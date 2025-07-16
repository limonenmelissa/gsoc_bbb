from Model_multi_te import deltaM_multite_model
from scipy.optimize import curve_fit
import numpy as np
import pymc

"""
Multi-TE ASL LM fitting

Input:
tis - inversion times (seconds)
tes - echo times (seconds) 
ntes - number of echo times per inversion time
signal - deltaM signal for a single voxel
m0a - Scaling factor: M0 value of the arterial blood signal
taus - Duration of the labelling phase (tau) [s] - can be array or scalar

Objective: Find parameters which fit the multi-TE deltaM signal best

att - Arterial transit time (ATT), time until the blood arrives in the voxel [s]
cbf - Cerebral blood flow (CBF) in [ml/min/100g]
texch - Exchange time between blood and tissue [s]
itt - Intra-voxel transit time [s]
"""


def fit_voxel_2params(tis, tes, ntes, signal, m0a, taus):
	"""
	2-parameter fitting with ATT and CBF as free parameters, other parameters fixed
	"""

	def model_func(dummy_var, att, cbf):
		# dummy_var seems to be required by curve_fit?
		return deltaM_multite_model(tis, tes, ntes, att, cbf, m0a, taus)

	# Initial parameter guess
	param0 = [1.2, 60]  # [att, cbf]

	# Parameter bounds
	bounds = ([0.1, 10], [2.5, 200])  # [att_min, cbf_min], [att_max, cbf_max]

	try:
		# Create dummy x data for curve_fit (not actually used in model)
		x_dummy = np.arange(len(signal))

		param_opt, param_cov = curve_fit(
			model_func, x_dummy, signal,
			p0=param0, bounds=bounds,
			maxfev=1000
		)

		# Calculate parameter uncertainties
		param_std = np.sqrt(np.diag(param_cov))

		return param_opt[0], param_opt[1], param_std[0], param_std[1]

	except (RuntimeError, ValueError) as e:
		return np.nan, np.nan, np.nan, np.nan


def fit_voxel_extended(tis, tes, ntes, signal, m0a, taus):
	"""
	Extended fitting with ATT, CBF, exchange time, and transit time as free parameters
	"""

	def model_func(dummy_var, att, cbf, texch, itt):
		return deltaM_multite_model(
			tis, tes, ntes, att, cbf, m0a, taus,
			texch=texch, itt=itt
		)

	# Initial parameter guess
	param0 = [1.2, 60, 0.1, 0.2]  # [att, cbf, texch, itt]

	# Parameter bounds
	bounds = (
		[0.1, 10, 0.01, 0.05],  # lower bounds
		[2.5, 200, 0.5, 0.8]  # upper bounds
	)

	try:
		# Create dummy x data for curve_fit
		x_dummy = np.arange(len(signal))

		param_opt, param_cov = curve_fit(
			model_func, x_dummy, signal,
			p0=param0, bounds=bounds,
			maxfev=2000
		)

		# Calculate parameter uncertainties
		param_std = np.sqrt(np.diag(param_cov))

		return (param_opt[0], param_opt[1], param_opt[2], param_opt[3],
				param_std[0], param_std[1], param_std[2], param_std[3])

	except (RuntimeError, ValueError) as e:
		return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def fit_volume_2params(pwi_data, tis, tes, ntes, m0_data, taus, lambd=0.9):
	"""
	Fit multi-TE model to whole volume with basic 2-parameter fitting (ATT, CBF)

	Parameters:
	- pwi_data: 4D array (x, y, z, time_points)
	- tis: array of inversion times
	- tes: array of echo times
	- ntes: array of number of TEs per TI
	- m0_data: 3D array of M0 values
	- taus: labeling duration(s)
	- lambd: blood-tissue partition coefficient
	"""
	shape = pwi_data.shape[:3]
	att_map = np.full(shape, np.nan)
	cbf_map = np.full(shape, np.nan)
	att_std_map = np.full(shape, np.nan)
	cbf_std_map = np.full(shape, np.nan)

	total_voxels = np.prod(shape)
	processed_voxels = 0

	for x in range(shape[0]):
		for y in range(shape[1]):
			for z in range(shape[2]):
				signal = pwi_data[x, y, z, :]

				# Skip invalid voxels
				if (np.any(np.isnan(signal)) or
						np.any(np.isinf(signal)) or
						np.all(signal == 0)):
					continue

				m0 = m0_data[x, y, z]

				if (np.isnan(m0) or np.isinf(m0) or m0 <= 0):
					continue

				# Calculate m0a scaling factor
				m0a = m0 / (6000 * lambd)

				try:
					att, cbf, att_std, cbf_std = fit_voxel_2params(
						tis, tes, ntes, signal, m0a, taus
					)

					att_map[x, y, z] = att
					cbf_map[x, y, z] = cbf
					att_std_map[x, y, z] = att_std
					cbf_std_map[x, y, z] = cbf_std

					processed_voxels += 1

				except Exception as e:
					continue

		# Progress update
		if x % 10 == 0:
			progress = (x * shape[1] * shape[2]) / total_voxels * 100
			print(f"Progress: {progress:.1f}% ({processed_voxels} voxels processed)")

	print(f"Fitting finished. {processed_voxels}/{total_voxels} voxels processed successfully.")

	return att_map, cbf_map, att_std_map, cbf_std_map


def fit_volume_extended(pwi_data, tis, tes, ntes, m0_data, taus, lambd=0.9):
	"""
	Fit multi-TE model to whole volume for 4-parameter fitting
	(ATT, CBF, exchange time, transit time)
	"""
	shape = pwi_data.shape[:3]
	att_map = np.full(shape, np.nan)
	cbf_map = np.full(shape, np.nan)
	texch_map = np.full(shape, np.nan)
	itt_map = np.full(shape, np.nan)

	att_std_map = np.full(shape, np.nan)
	cbf_std_map = np.full(shape, np.nan)
	texch_std_map = np.full(shape, np.nan)
	itt_std_map = np.full(shape, np.nan)

	total_voxels = np.prod(shape)
	processed_voxels = 0

	for x in range(shape[0]):
		for y in range(shape[1]):
			for z in range(shape[2]):
				signal = pwi_data[x, y, z, :]

				# Skip invalid voxels
				if (np.any(np.isnan(signal)) or
						np.any(np.isinf(signal)) or
						np.all(signal == 0)):
					continue

				m0 = m0_data[x, y, z]

				if (np.isnan(m0) or np.isinf(m0) or m0 <= 0):
					continue

				# Calculate m0a scaling factor
				m0a = m0 / (6000 * lambd)

				try:
					(att, cbf, texch, itt,
					 att_std, cbf_std, texch_std, itt_std) = fit_voxel_extended(
						tis, tes, ntes, signal, m0a, taus
					)

					att_map[x, y, z] = att
					cbf_map[x, y, z] = cbf
					texch_map[x, y, z] = texch
					itt_map[x, y, z] = itt

					att_std_map[x, y, z] = att_std
					cbf_std_map[x, y, z] = cbf_std
					texch_std_map[x, y, z] = texch_std
					itt_std_map[x, y, z] = itt_std

					processed_voxels += 1

				except Exception as e:
					continue

		# Update the progress
		if x % 10 == 0:
			progress = (x * shape[1] * shape[2]) / total_voxels * 100
			print(f"Progress: {progress:.1f}% ({processed_voxels} voxels processed)")

	print(f"Extended fitting completed. {processed_voxels}/{total_voxels} voxels processed successfully.")

	return (att_map, cbf_map, texch_map, itt_map,
			att_std_map, cbf_std_map, texch_std_map, itt_std_map)


