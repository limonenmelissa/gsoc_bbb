import numpy as np

"""
Multi-TE ASL Model based on the paper
Robust Multi-TE ASL-Based Blood–Brain Barrier Integrity Measurements. Front Neuroscience 15. (2021)
Implements the three-compartment model (S_bl1, S_bl2, S_ex) with T1 and T2 decay
"""

def deltaM_multite_model(tis, tes, ntes, att, cbf, m0a, taus,
						 t1=1.3, t1b=1.65, t2=0.050, t2b=0.150,
						 texch=0.1, itt=0.2, lambd=0.9, alpha=0.68):
	"""
	Parameters:
	- tis: array of inversion times [s]
	- tes: array of echo times [s]
	- ntes: array of number of TEs per TI
	- att: arterial transit time [s]
	- cbf: cerebral blood flow  [ml/min/100g]
	- m0a: scaling factor
	- taus: array or single value of labeling durations per TI [s]
	- t1: T1 of tissue [s]
	- t1b: T1 of blood [s]
	- t2: T2 of tissue [s]
	- t2b: T2 of blood [s]
	- texch: exchange time [s]
	- itt: intra-voxel transit time [s]
	- lambd: blood-tissue partition coefficient
	- alpha: labeling efficiency

	Returns:
	- deltaM: signal for all TE/TI combinations
	"""

	# Convert CBF from ml/min/100g to ml/s/g
	# 60s -> 1s, 100g -> 1g,  1/6000 similar to scaling in one compartment model (see voxelwise_model.py)
	f = cbf / 100.0 * 60.0 / 6000.0

	# Handle taus input - can be single value or array
	if np.isscalar(taus):
		tau_array = np.full(len(tis), taus)
	else:
		tau_array = np.array(taus)

	# Initialize result arrays
	S_bl1_final = np.zeros(len(tes))
	S_bl2_final = np.zeros(len(tes))
	S_ex_final = np.zeros(len(tes))

	te_index = 0

	# Loop over all inversion times
	# C++: for (int j = 0; j < tis.size(); ++j)

	for j in range(len(tis)):
		tau = tau_array[j]
		ti = tis[j]

		# Case 1: 0 < ti < att
		if 0 < ti < att:
			for k in range(ntes[j]):
				S_bl1_final[te_index] = 0.0
				S_bl2_final[te_index] = 0.0
				S_ex_final[te_index] = 0.0
				te_index += 1

		# Case 2: att <= ti < (att + itt)
		elif att <= ti < (att + itt):
			for k in range(ntes[j]):
				te = tes[te_index]

				# Subcase 1: 0 <= te < (att+itt-ti)
				if 0 <= te < (att + itt - ti):
					S_bl1_final[te_index] = (2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) *
											 (np.exp(ti / t1b) - np.exp(att / t1b)) * np.exp(-te / t2b))

					S_bl2_final[te_index] = 0.0

					S_ex_final[te_index] = 0.0

				# Subcase 2: (att+itt-ti) <= te < itt
				elif (att + itt - ti) <= te < itt:
					base_term = 2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) * (
								np.exp(ti / t1b) - np.exp(att / t1b))
					transition_factor = (te - (att + itt - ti)) / (ti - att)

					S_bl1_final[te_index] = ((base_term - transition_factor * base_term) * np.exp(-te / t2b))

					S_bl2_final[te_index] = (transition_factor * base_term * np.exp(-te / t2b) * np.exp(-te / texch))

					S_ex_final[te_index] = (
								transition_factor * base_term * (1 - np.exp(-te / texch)) * np.exp(-te / t2))

				# Subcase 3: te >= itt
				else:
					S_bl1_final[te_index] = 0.0

					S_bl2_final[te_index] = (2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) *
											 (np.exp(ti / t1b) - np.exp(att / t1b)) * np.exp(-te / t2b) * np.exp(
								-te / texch))

					S_ex_final[te_index] = (2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) *
											(np.exp(ti / t1b) - np.exp(att / t1b)) * (1 - np.exp(-te / texch)) * np.exp(
								-te / t2))

				te_index += 1

		# Case 3: (att+itt) <= ti < (att + tau)
		elif (att + itt) <= ti < (att + tau):
			for k in range(ntes[j]):
				te = tes[te_index]

				# Subcase 1: 0 <= te < itt
				if 0 <= te < itt:
					term1 = 2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) * (
								np.exp(ti / t1b) - np.exp(att / t1b))
					term2 = 2 * f * t1b * np.exp(-(att + itt) / t1b) * np.exp(-ti / t1b) * (
								np.exp(ti / t1b) - np.exp((att + itt) / t1b))
					base_diff = term1 - term2
					transition_factor = te / itt

					S_bl1_final[te_index] = ((base_diff - transition_factor * base_diff) * np.exp(-te / t2b))

					S_bl2_final[te_index] = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
											  np.exp(-((1 / t1b) + (1 / texch)) * ti) *
											  (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
												  ((1 / t1b) + (1 / texch)) * (att + itt))) +
											  transition_factor * base_diff) * np.exp(-te / t2b) * np.exp(-te / texch))

					ex_term1 = ((2 * f * np.exp(-(1 / t1b) * (att + itt))) / (1 / t1) * np.exp(-(1 / t1) * ti) *
								(np.exp((1 / t1) * ti) - np.exp((1 / t1) * (att + itt))) -
								(2 * f * np.exp(-(1 / t1b) * (att + itt))) / ((1 / texch) + (1 / t1)) *
								np.exp(-((1 / t1) + (1 / texch)) * ti) *
								(np.exp(((1 / texch) + (1 / t1)) * ti) - np.exp(
									((1 / texch) + (1 / t1)) * (att + itt)))) * np.exp(-te / t2)

					ex_term2 = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
								 np.exp(-((1 / t1b) + (1 / texch)) * ti) *
								 (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
									 ((1 / t1b) + (1 / texch)) * (att + itt))) +
								 transition_factor * base_diff) * (1 - np.exp(-te / texch)) * np.exp(-te / t2))

					S_ex_final[te_index] = ex_term1 + ex_term2

				# Subcase 2: te >= itt
				else:
					term1 = 2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) * (
								np.exp(ti / t1b) - np.exp(att / t1b))
					term2 = 2 * f * t1b * np.exp(-(att + itt) / t1b) * np.exp(-ti / t1b) * (
								np.exp(ti / t1b) - np.exp((att + itt) / t1b))

					S_bl1_final[te_index] = 0.0

					S_bl2_final[te_index] = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
											  np.exp(-((1 / t1b) + (1 / texch)) * ti) *
											  (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
												  ((1 / t1b) + (1 / texch)) * (att + itt))) +
											  (term1 - term2)) * np.exp(-te / t2b) * np.exp(-te / texch))

					ex_term1 = ((2 * f * np.exp(-(1 / t1b) * (att + itt))) / (1 / t1) * np.exp(-(1 / t1) * ti) *
								(np.exp((1 / t1) * ti) - np.exp((1 / t1) * (att + itt))) -
								(2 * f * np.exp(-(1 / t1b) * (att + itt))) / ((1 / texch) + (1 / t1)) *
								np.exp(-((1 / t1) + (1 / texch)) * ti) *
								(np.exp(((1 / texch) + (1 / t1)) * ti) - np.exp(
									((1 / texch) + (1 / t1)) * (att + itt)))) * np.exp(-te / t2)

					ex_term2 = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
								 np.exp(-((1 / t1b) + (1 / texch)) * ti) *
								 (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
									 ((1 / t1b) + (1 / texch)) * (att + itt))) +
								 (term1 - term2)) * (1 - np.exp(-te / texch)) * np.exp(-te / t2))

					S_ex_final[te_index] = ex_term1 + ex_term2

				te_index += 1

		# Case 4: (att + tau) <= ti < (att + itt + tau)
		elif (att + tau) <= ti < (att + itt + tau):
			for k in range(ntes[j]):
				te = tes[te_index]

				# Subcase 1: 0 <= te < (itt-(ti-(att+tau)))
				if 0 <= te < (itt - (ti - (att + tau))):
					term1 = 2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) * (
								np.exp((att + tau) / t1b) - np.exp(att / t1b))
					term2 = 2 * f * t1b * np.exp(-(att + itt) / t1b) * np.exp(-ti / t1b) * (
								np.exp(ti / t1b) - np.exp((att + itt) / t1b))
					base_diff = term1 - term2
					transition_factor = te / (itt - (ti - (att + tau)))

					S_bl1_final[te_index] = ((base_diff - transition_factor * base_diff) * np.exp(-te / t2b))

					S_bl2_final[te_index] = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
											  np.exp(-((1 / t1b) + (1 / texch)) * ti) *
											  (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
												  ((1 / t1b) + (1 / texch)) * (att + itt))) +
											  transition_factor * base_diff) * np.exp(-te / t2b) * np.exp(-te / texch))

					ex_term1 = ((2 * f * np.exp(-(1 / t1b) * (att + itt))) / (1 / t1) * np.exp(-(1 / t1) * ti) *
								(np.exp((1 / t1) * ti) - np.exp((1 / t1) * (att + itt))) -
								(2 * f * np.exp(-(1 / t1b) * (att + itt))) / ((1 / texch) + (1 / t1)) *
								np.exp(-((1 / t1) + (1 / texch)) * ti) *
								(np.exp(((1 / texch) + (1 / t1)) * ti) - np.exp(
									((1 / texch) + (1 / t1)) * (att + itt)))) * np.exp(-te / t2)

					ex_term2 = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
								 np.exp(-((1 / t1b) + (1 / texch)) * ti) *
								 (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
									 ((1 / t1b) + (1 / texch)) * (att + itt))) +
								 transition_factor * base_diff) * (1 - np.exp(-te / texch)) * np.exp(-te / t2))

					S_ex_final[te_index] = ex_term1 + ex_term2

				# Subcase 2: te >= (itt-(ti-(att+tau)))
				else:
					term1 = 2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) * (
								np.exp((att + tau) / t1b) - np.exp(att / t1b))
					term2 = 2 * f * t1b * np.exp(-(att + itt) / t1b) * np.exp(-ti / t1b) * (
								np.exp(ti / t1b) - np.exp((att + itt) / t1b))

					S_bl1_final[te_index] = 0.0

					S_bl2_final[te_index] = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
											  np.exp(-((1 / t1b) + (1 / texch)) * ti) *
											  (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
												  ((1 / t1b) + (1 / texch)) * (att + itt))) +
											  (term1 - term2)) * np.exp(-te / t2b) * np.exp(-te / texch))

					ex_term1 = ((2 * f * np.exp(-(1 / t1b) * (att + itt))) / (1 / t1) * np.exp(-(1 / t1) * ti) *
								(np.exp((1 / t1) * ti) - np.exp((1 / t1) * (att + itt))) -
								(2 * f * np.exp(-(1 / t1b) * (att + itt))) / ((1 / texch) + (1 / t1)) *
								np.exp(-((1 / t1) + (1 / texch)) * ti) *
								(np.exp(((1 / texch) + (1 / t1)) * ti) - np.exp(
									((1 / texch) + (1 / t1)) * (att + itt)))) * np.exp(-te / t2)

					ex_term2 = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
								 np.exp(-((1 / t1b) + (1 / texch)) * ti) *
								 (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
									 ((1 / t1b) + (1 / texch)) * (att + itt))) +
								 (term1 - term2)) * (1 - np.exp(-te / texch)) * np.exp(-te / t2))

					S_ex_final[te_index] = ex_term1 + ex_term2

				te_index += 1

		# Case 5: ti >= (att + itt + tau)
		else:
			for k in range(ntes[j]):
				te = tes[te_index]

				S_bl1_final[te_index] = 0.0

				S_bl2_final[te_index] = (2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
										 np.exp(-((1 / t1b) + (1 / texch)) * ti) *
										 (np.exp(((1 / t1b) + (1 / texch)) * (att + itt + tau)) - np.exp(
											 ((1 / t1b) + (1 / texch)) * (att + itt))) *
										 np.exp(-te / t2b) * np.exp(-te / texch))

				ex_term1 = ((2 * f * np.exp(-(1 / t1b) * (att + itt))) / (1 / t1) * np.exp(-(1 / t1) * ti) *
							(np.exp((1 / t1) * (att + itt + tau)) - np.exp((1 / t1) * (att + itt))) -
							(2 * f * np.exp(-(1 / t1b) * (att + itt))) / ((1 / texch) + (1 / t1)) *
							np.exp(-((1 / t1) + (1 / texch)) * ti) *
							(np.exp(((1 / texch) + (1 / t1)) * (att + itt + tau)) - np.exp(
								((1 / texch) + (1 / t1)) * (att + itt)))) * np.exp(-te / t2)

				ex_term2 = ((2 * f * np.exp(-((1 / t1b)) * (att + itt)) / (((1 / t1b)) + (1 / texch)) *
							 np.exp(-(((1 / t1b)) + (1 / texch)) * ti) *
							 (np.exp(((1 / t1b) + (1 / texch)) * (att + itt + tau)) - np.exp(
								 ((1 / t1b) + (1 / texch)) * (att + itt)))) *
							(1 - np.exp(-te / texch)) * np.exp(-te / t2))

				S_ex_final[te_index] = ex_term1 + ex_term2

				te_index += 1

	# Sum all three compartments to get final signal
	delta_M_final = S_bl1_final + S_bl2_final + S_ex_final

	# Apply scaling factors
	delta_M_final *= m0a * alpha / lambd

	return delta_M_final
