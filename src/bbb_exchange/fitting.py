from voxelwise_model import deltaM_model
from scipy.optimize import curve_fit
import numpy as np
import pymc


"""
Input:
t - time points (seconds)
signal - deltaM time period for a single voxel
m0a - Scaling factor: M0 value of the arterial blood signal (e.g. from M0. nii)
tau - Duration of the labelling phase (tau) [s]

Objective: Find two parameters ATT and CBF which fit the deltaM time signal best
att - Arterial transit time (ATT), time until the blood arrives in the voxel [s]
cbf - Cerebral blood flow (CBF) in [ml/min/100g]
"""


def fit_voxel(t, signal, m0a, tau):
	def model_func(t, att, cbf):
		return deltaM_model(t, att, cbf, m0a, tau)

	param0 = [1.2, 60]
	bounds = ([0.1, 10], [1.5, 200])

	try:
		param_opt, _ = curve_fit(model_func, t, signal, p0=param0, bounds=bounds)
		return param_opt[0], param_opt[1]
	except RuntimeError:
		return np.nan, np.nan


def fit_volume(pwi_data, t, m0_data, tau, lambd=0.9):

	shape = pwi_data.shape[:3]
	att_map = np.full(shape, np.nan)
	cbf_map = np.full(shape, np.nan)

	for x in range(shape[0]):
		for y in range(shape[1]):
			for z in range(shape[2]):
				signal = pwi_data[x, y, z, :]

				if (
						np.any(np.isnan(signal)) or
						np.any(np.isinf(signal)) or
						np.all(signal == 0)
				):
					continue

				m0 = m0_data[x, y, z]

				if (
						np.isnan(m0) or
						np.isinf(m0) or
						m0 <= 0
				):
					continue

				m0a = m0 / (6000 * lambd)

				try:
					att, cbf = fit_voxel(t, signal, m0a, tau)
					att_map[x, y, z] = att
					cbf_map[x, y, z] = cbf
				except Exception as e:
					continue

	return att_map, cbf_map



def bayesian_fit_voxel(t, signal, m0a, tau):
    t = np.array(t, dtype=np.float32)

    with pymc.Model():
        att = pymc.Normal("ATT", mu=1.2, sigma=0.3)
        cbf = pymc.Normal("CBF", mu=60, sigma=15)

        mu = deltaM_model(t, att, cbf, m0a, tau)

        sigma_val = np.std(signal)
        sigma = pymc.HalfNormal("sigma", sigma=max(sigma_val, 0.1))
        pymc.Normal("obs", mu=mu, sigma=sigma, observed=signal)

        trace = pymc.sample(300, tune=200, chains=2, progressbar=True, return_inferencedata=True)

    return trace


def bayesian_fit_volume(pwi_data, t, m0_data, tau, lambda_blood=0.9):
    shape = pwi_data.shape[:3]
    att_map = np.full(shape, np.nan)
    cbf_map = np.full(shape, np.nan)

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                signal = pwi_data[x, y, z, :]

                if (
                    np.any(np.isnan(signal)) or
                    np.any(np.isinf(signal)) or
                    np.all(signal == 0)
                ):
                    continue

                m0 = m0_data[x, y, z]

                if (
                    np.isnan(m0) or
                    np.isinf(m0) or
                    m0 <= 0
                ):
                    continue

                m0a = m0 / (6000 * lambda_blood)
                try:
                    trace = bayesian_fit_voxel(t, signal, m0a, tau)
                    if trace is None:
                        continue

                    att_samples = trace.posterior["ATT"].values.flatten()
                    cbf_samples = trace.posterior["CBF"].values.flatten()

                    att_map[x, y, z] = np.mean(att_samples)
                    cbf_map[x, y, z] = np.mean(cbf_samples)

                except Exception as e:
                    continue

    return att_map, cbf_map