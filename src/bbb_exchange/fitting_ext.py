from voxelwise_model_ext import deltaM_model_ext
from scipy.optimize import curve_fit
import numpy as np
import pymc

"""
Input:
t - time points (seconds)
signal - deltaM time period for a single voxel
m0a - Scaling factor: M0 value of the arterial blood signal (e.g. from M0. nii)
tau - Duration of the labelling phase (tau) [s]

Objective: Fit ATT, CBF and aBV for extended model (tissue + arterial compartments)
"""

def fit_voxel(t, signal, m0a, tau):
    def model_func(t, att, cbf, abv):
        return deltaM_model_ext(t, att, cbf, m0a, tau, abv=abv)

    param0 = [1.2, 60, 0.01]
    bounds = ([0.1, 10, 0.0], [1.5, 200, 0.2])

    try:
        param_opt, _ = curve_fit(model_func, t, signal, p0=param0, bounds=bounds)
        return param_opt[0], param_opt[1], param_opt[2]
    except RuntimeError:
        return np.nan, np.nan, np.nan

def fit_volume(pwi_data, t, m0_data, tau, lambd=0.9):
    shape = pwi_data.shape[:3]
    att_map = np.full(shape, np.nan)
    cbf_map = np.full(shape, np.nan)
    abv_map = np.full(shape, np.nan)

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
                    att, cbf, abv = fit_voxel(t, signal, m0a, tau)
                    att_map[x, y, z] = att
                    cbf_map[x, y, z] = cbf
                    abv_map[x, y, z] = abv
                except Exception:
                    continue

    return att_map, cbf_map, abv_map

def bayesian_fit_voxel(t, signal, m0a, tau):
    t = np.array(t, dtype=np.float32)

    with pymc.Model():
        att = pymc.Normal("ATT", mu=1.2, sigma=0.3)
        cbf = pymc.Normal("CBF", mu=60, sigma=15)
        abv = pymc.HalfNormal("aBV", sigma=0.05)

        mu = deltaM_model_ext(t, att, cbf, m0a, tau, abv=abv)

        sigma_val = np.std(signal)
        sigma = pymc.HalfNormal("sigma", sigma=max(sigma_val, 0.1))
        pymc.Normal("obs", mu=mu, sigma=sigma, observed=signal)

        trace = pymc.sample(300, tune=200, chains=2, progressbar=True, return_inferencedata=True)

    return trace

def bayesian_fit_volume(pwi_data, t, m0_data, tau, lambda_blood=0.9):
    shape = pwi_data.shape[:3]
    att_map = np.full(shape, np.nan)
    cbf_map = np.full(shape, np.nan)
    abv_map = np.full(shape, np.nan)

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
                    abv_samples = trace.posterior["aBV"].values.flatten()

                    att_map[x, y, z] = np.mean(att_samples)
                    cbf_map[x, y, z] = np.mean(cbf_samples)
                    abv_map[x, y, z] = np.mean(abv_samples)

                except Exception:
                    continue

    return att_map, cbf_map, abv_map
