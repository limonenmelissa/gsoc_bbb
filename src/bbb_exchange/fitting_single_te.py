import numpy as np
import stan
from scipy.optimize import curve_fit
from deltaM_model import deltaM_model

"""
Objective: Find two parameters ATT and CBF which fit the deltaM time signal best
att - Arterial transit time [s]
cbf - Cerebral blood flow in [ml/min/100g]

Input:
t - time points [s]
signal - deltaM time period for a single voxel
m0a - Scaling factor: M0 value of the arterial blood signal (e.g. from M0.nii)
tau - Duration of the labelling phase [s]
"""

# Data fitting is done using the stan package, therefore the model equation from deltaM_model.py must be rewritten
STAN_MODEL_CODE = """
functions {
  vector deltaM_model(vector t, real att, real cbf, real m0a, real tau, real T1a, real lambda_val, real alpha) {
    int n = num_elements(t);
    vector[n] deltaM = rep_vector(0.0, n);

    for (i in 1:n) {
      if (t[i] > att) {
        real exp1 = exp(-(t[i] - att) / T1a);
        real exp2 = exp(-(t[i] - att - tau) / T1a);
        deltaM[i] = (2 * alpha * cbf / 6000) * m0a * exp(-att / T1a) * (T1a / lambda_val) * (exp1 - exp2);
      }
    }
    return deltaM;
  }
}

data {
  int<lower=0> n;
  vector[n] t;
  vector[n] signal;
  real<lower=0> m0a;
  real<lower=0> tau;
  real<lower=0> T1a;
  real<lower=0> lambda_val;
  real<lower=0> alpha;
}

parameters {
  real<lower=0.1, upper=3.0> att;
  real<lower=10, upper=200> cbf;
  real<lower=0.001> sigma;
}

model {
  vector[n] mu;

  // Priors
  att ~ normal(1.2, 0.3);
  cbf ~ normal(60, 15);
  sigma ~ exponential(1);

  // Likelihood
  mu = deltaM_model(t, att, cbf, m0a, tau, T1a, lambda_val, alpha);
  signal ~ normal(mu, sigma);
}
"""


def ls_fit_voxel(t, signal, m0a, tau):
	"""
	Least squares fitting for a single voxel
	"""

	def model_func(t, att, cbf):
		return deltaM_model(t, att, cbf, m0a, tau)

	param0 = [1.2, 60]
	bounds = ([0.1, 10], [1.5, 200])

	try:
		param_opt, _ = curve_fit(model_func, t, signal, p0=param0, bounds=bounds)
		return param_opt[0], param_opt[1]
	except RuntimeError:
		return np.nan, np.nan


def ls_fit_volume(pwi_data, t, m0_data, tau, lambd=0.9):
	"""
	Least squares fitting for entire volume
	"""
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
					att, cbf = ls_fit_voxel(t, signal, m0a, tau)
					att_map[x, y, z] = att
					cbf_map[x, y, z] = cbf
				except Exception as e:
					continue

	return att_map, cbf_map


def bayesian_fit_voxel(t, signal, m0a, tau, T1a=1.65, lambd=0.9, alpha=0.85 * 0.8):
	"""
	Bayesian fitting for a single voxel
	"""
	# Data handling for using Stan
	data = {
		'n': len(t),
		't': t.astype(float),
		'signal': signal.astype(float),
		'm0a': float(m0a),
		'tau': float(tau),
		'T1a': float(T1a),
		'lambda_val': float(lambd),
		'alpha': float(alpha)
	}

	try:
		# Build the model
		model = stan.build(STAN_MODEL_CODE, data=data)

		# Sample from the posterior
		fit = model.sample(num_chains=2, num_samples=1000, num_warmup=500)

		return fit

	except Exception as e:
		print(f"Error in Stan fitting: {e}")
		return None


def bayesian_fit_volume(pwi_data, t, m0_data, tau, lambda_blood=0.9, T1a=1.65, alpha=0.85 * 0.8):
	"""
	Bayesian fitting for entire volume
	"""
	shape = pwi_data.shape[:3]
	att_map = np.full(shape, np.nan)
	cbf_map = np.full(shape, np.nan)
	att_std_map = np.full(shape, np.nan)
	cbf_std_map = np.full(shape, np.nan)

	total_voxels = shape[0] * shape[1] * shape[2]
	processed_voxels = 0

	for x in range(shape[0]):
		for y in range(shape[1]):
			for z in range(shape[2]):
				processed_voxels += 1
				if processed_voxels % 100 == 0:
					print(f"Processing voxel {processed_voxels}/{total_voxels}")

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
					fit = bayesian_fit_voxel(t, signal, m0a, tau, T1a, lambda_blood, alpha)
					if fit is None:
						continue

					# Posterior samples
					att_samples = fit['att'].flatten()
					cbf_samples = fit['cbf'].flatten()

					att_map[x, y, z] = np.mean(att_samples)
					cbf_map[x, y, z] = np.mean(cbf_samples)
					att_std_map[x, y, z] = np.std(att_samples)
					cbf_std_map[x, y, z] = np.std(cbf_samples)

				except Exception as e:
					print(f"Error fitting voxel ({x}, {y}, {z}): {e}")
					continue

	return att_map, cbf_map, att_std_map, cbf_std_map


def extract_posterior_summary(fit):
	"""
	Extract summary statistics
	"""
	if fit is None:
		return None

	try:
		att_samples = fit['att'].flatten()
		cbf_samples = fit['cbf'].flatten()

		summary = {
			'att_mean': np.mean(att_samples),
			'att_std': np.std(att_samples),
			'cbf_mean': np.mean(cbf_samples),
			'cbf_std': np.std(cbf_samples)
		}

		return summary

	except Exception as e:
		print(f"Error extracting posterior summary: {e}")
		return None
