import numpy as np
import matplotlib.pyplot as plt

def deltaM_model(t, att, cbf, m0a, tau, T1a=1.65, lambd=0.9, alpha=0.85 * 0.8):
	"""
	Modelling of the ASL signal (DeltaM) over time using Chappell equation [1] from the paper 'Separation of macrovascular signal in multi‐inversion time arterial spin labelling MRI', https://doi.org/10.1002/mrm.22320

	Inputs:
	 t - Time points (seconds)
	 att - Arterial transit time (ATT), time until the blood arrives in the voxel [s]
	 cbf - Cerebral blood flow (CBF) in [ml/min/100g]
	 m0a - Scaling factor: M0 value of the arterial blood signal (e.g. from M0. nii)
	 tau - Duration of the labelling phase (tau) [s]
	 T1a - Longitudinal relaxation time of the arterial blood [s] (default: 1.65s)
	 lambd - Blood-tissue-water partition coefficient (default: 0.9)
	 alpha - Effective labelling efficiency (e.g. 0.85 * 0.8)
	 Output:
	 deltaM - Expected signal curve DeltaM(t) for a single voxel
	"""
	# Initialise time array with zeros
	deltaM = np.zeros_like(t)

	cbf_factor = 2 * alpha * cbf / 6000  # Conversion factor: CBF from ml/min/100g → ml/s/g
	m0_factor = m0a * np.exp(-att / T1a)  # Scaling by M0a
	t1_lambda_factor = T1a / lambd  # Normalisation to tissue water concentration

	# During bolus: att < t <= att + tau
	during_bolus = (t > att) & (t <= att + tau)
	if np.any(during_bolus):
		t_bolus = t[during_bolus]
		exp_term = 1 - np.exp(-(t_bolus - att) / T1a)
		deltaM[during_bolus] = cbf_factor * m0_factor * t1_lambda_factor * exp_term

	# After bolus: t > att + tau
	after_bolus = t > att + tau
	if np.any(after_bolus):
		t_after = t[after_bolus]
		exp1 = np.exp(-(t_after - att) / T1a)
		exp2 = np.exp(-(t_after - att - tau) / T1a)
		deltaM[after_bolus] = cbf_factor * m0_factor * t1_lambda_factor * (exp1 - exp2)

	return deltaM


def plot_asl_signal():
	"""
	Plot of ASL signal
	"""
	# define time points
	t = np.linspace(0, 8, 800)

	# set parameters
	att = 1.2
	cbf = 60
	m0a = 3000
	tau = 1.8
	T1a = 1.65
	alpha = 0.85 * 0.8

	# calculate signal with model equation
	deltaM = deltaM_model(t, att, cbf, m0a, tau, T1a, alpha=alpha)

	# create plot
	plt.figure(figsize=(10, 6))
	plt.plot(t, deltaM, 'b-', linewidth=2, label='DeltaM(t)')

	# vertical lines for ATT und ATT+tau
	plt.axvline(x=att, color='red', linestyle='--', alpha=0.7, label=f'ATT = {att}s')
	plt.axvline(x=att + tau, color='orange', linestyle='--', alpha=0.7, label=f'ATT+τ = {att + tau}s')

	# Labelling
	plt.xlabel('Time [s]', fontsize=12)
	plt.ylabel('DeltaM(t) [Signal Intensity]', fontsize=12)
	plt.title('ASL Signal - Chappell Model', fontsize=14, fontweight='bold')
	plt.grid(True, alpha=0.3)
	plt.legend()
	param_text = f'ATT = {att}s\nCBF = {cbf} ml/min/100g\nτ = {tau}s\nM0a = {m0a}\nα = {alpha:.2f}'
	plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes,
			 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

	plt.tight_layout()
	plt.show()



def plot_parameter_sensitivity():
	"""
    Comparing output for different parameter settings
    """
	t = np.linspace(0, 8, 800)
	base_params = {'att': 1.0, 'cbf': 50, 'tau': 1.5, 'm0a': 3000, 'T1a': 1.65, 'alpha': 0.68}

	fig, axes = plt.subplots(2, 2, figsize=(15, 10))
	axes = axes.flatten()

	# ATT sensitivity
	ax = axes[0]
	for att in [0.5, 1.0, 1.5, 2.0]:
		params = base_params.copy()
		params['att'] = att
		deltaM = deltaM_model(t, **params)
		ax.plot(t, deltaM, linewidth=2, label=f'ATT = {att}s')
	ax.set_title('ATT sensitivity')
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('DeltaM(t)')
	ax.legend()
	ax.grid(True, alpha=0.3)

	# CBF sensitivity
	ax = axes[1]
	for cbf in [25, 50, 75, 100]:
		params = base_params.copy()
		params['cbf'] = cbf
		deltaM = deltaM_model(t, **params)
		ax.plot(t, deltaM, linewidth=2, label=f'CBF = {cbf}')
	ax.set_title('CBF sensitivity')
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('DeltaM(t)')
	ax.legend()
	ax.grid(True, alpha=0.3)

	# Tau sensifivity
	ax = axes[2]
	for tau in [0.5, 1.0, 1.5, 2.0]:
		params = base_params.copy()
		params['tau'] = tau
		deltaM = deltaM_model(t, **params)
		ax.plot(t, deltaM, linewidth=2, label=f'τ = {tau}s')
	ax.set_title('Tau sensitivity')
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('DeltaM(t)')
	ax.legend()
	ax.grid(True, alpha=0.3)

	# Alpha sensitivity
	ax = axes[3]
	for alpha in [0.4, 0.55, 0.68, 0.8]:
		params = base_params.copy()
		params['alpha'] = alpha
		deltaM = deltaM_model(t, **params)
		ax.plot(t, deltaM, linewidth=2, label=f'α = {alpha:.2f}')
	ax.set_title('Alpha sensitivity')
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('DeltaM(t)')
	ax.legend()
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":

	plot_asl_signal()
	plot_parameter_sensitivity()
