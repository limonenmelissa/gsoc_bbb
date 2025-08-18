import numpy as np
import matplotlib.pyplot as plt


def dm_tiss(t, Dt, tau, f, M0a, a, T1, T1a, k=0.9):
	"""
        Chappell equation [1] (tissue component) and [2] (arterial component) from the paper 'Separation of macrovascular signal in multi‐inversion time arterial spin labelling MRI', https://doi.org/10.1002/mrm.22320
        Modelling of the ASL signal (DeltaM) over time.
        Inputs:
             t - Time points (seconds)
             att - Arterial transit time (Dt in paper), time until the blood arrives in the voxel [s]
             cbf - Cerebral blood flow (CBF) in [ml/min/100g]
             m0a - Scaling factor: M0 value of the arterial blood signal (e.g. from M0. nii)
             tau - Duration of the labelling phase (tau) [s]
             T1a - Longitudinal relaxation time of the arterial blood [s] (default: 1.65s)
             lambd - Blood-tissue-water partition coefficient (default: 0.9)
             alpha - Labelling efficiency (e.g. 0.85 * 0.8)
         Output:
         deltaM - Expected signal curve DeltaM(t) for a single voxel
    """
	t = np.asarray(t, dtype=float)
	T1app = 1.0 / T1 + f / k
	R = 1.0 / T1app - 1.0 / T1a

	DM = np.zeros_like(t)

	# case 2: Dt <= t <= Dt + tau
	mask2 = (t >= Dt) & (t <= Dt + tau)
	if np.any(mask2):
		tt = t[mask2]
		term = (np.exp(R * tt) - np.exp(R * Dt))
		DM[mask2] = (2 * a * M0a * f * np.exp(-tt / T1app) / R) * term

	# case 3: t > Dt + tau
	mask3 = t > Dt + tau
	if np.any(mask3):
		tt = t[mask3]
		term = (np.exp(R * (Dt + tau)) - np.exp(R * Dt))
		DM[mask3] = (2 * a * M0a * f * np.exp(-tt / T1app) / R) * term

	return DM


def dm_art(t, Dta, ta, aBV, M0a, a, T1a):
	"""
        Chappell equation [2] (arterial component) from the paper 'Separation of macrovascular signal in multi‐inversion time arterial spin labelling MRI', https://doi.org/10.1002/mrm.22320
        Modelling of the ASL signal (DeltaM) over time.
        Inputs:
             t - Time points (seconds)
             att - Arterial transit time (Dt in paper), time until the blood arrives in the voxel [s]
             cbf - Cerebral blood flow (CBF) in [ml/min/100g]
             m0a - Scaling factor: M0 value of the arterial blood signal (e.g. from M0. nii)
             tau - Duration of the labelling phase (tau) [s]
             T1a - Longitudinal relaxation time of the arterial blood [s] (default: 1.65s)
             Dta - arterial arrival time (s)
             ta - artieral bolus time (s)
             abV - arterial blood volume
             lambd - Blood-tissue-water partition coefficient (default: 0.9)
             alpha - Labelling efficiency (e.g. 0.85 * 0.8)
         Output:
         deltaM - Expected signal curve DeltaM(t) for a single voxel
    """
	t = np.asarray(t, dtype=float)
	DM = np.zeros_like(t)
	# Dta <= t <= Dta + ta, only if signal exists
	mask = (t >= Dta) & (t <= Dta + ta)
	if np.any(mask):
		tt = t[mask]
		DM[mask] = 2 * a * M0a * aBV * np.exp(-tt / T1a)
	return DM


def DeltaM_model_ext(t, params):

	return dm_tiss(t, params['Dt'], params['tau'], params['f'], params['M0a'],
				   params['a'], params['T1'], params['T1a'], params.get('k', 0.9)) + \
		dm_art(t, params['Dta'], params['ta'], params['aBV'], params['M0a'],
			   params['a'], params['T1a'])


t = np.linspace(0, 3.0, 301)

params = {
	'f': 0.01,  # ml/g/s
	'Dt': 0.7,  # s
	'tau': 1.0,  # s
	'M0a': 1.0,  # scaling
	'a': 0.95,  # efficiency factor
	'T1': 1.3,  # s
	'T1a': 1.6,  # s
	'k': 0.9, # lambda
	'aBV': 0.01,  # arterial blood volume fraction
	'Dta': 0.5,  # arterial arrival time
	'ta': 1.0  # arteral bolus time
}

"""
# Visualising the model
Dm_tissue = dm_tiss(t, params['Dt'], params['tau'], params['f'], params['M0a'],
					params['a'], params['T1'], params['T1a'], params['k'])
Dm_art = dm_art(t, params['Dta'], params['ta'], params['aBV'], params['M0a'],
				params['a'], params['T1a'])
Dm_tot = Dm_tissue + Dm_art


plt.figure(figsize=(8, 4))
plt.plot(t, Dm_tot, label='DM_total (tissue + arterial)')
plt.plot(t, Dm_tissue, '--', label='tissue component')
plt.plot(t, Dm_art, ':', label='arterial component')
plt.xlabel('time (s)')
plt.ylabel('DeltaM')
plt.title('Simulated ASL kinetic curves (Equation 1 + 2)')
plt.legend()
plt.tight_layout()
plt.show()
"""
