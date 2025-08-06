import numpy as np
import os
from data_handling import load_nifti_file
import stan
from deltaM_model import deltaM_model
from scipy.optimize import curve_fit

"""
This file provides the necessary functions to
1. perform least squares fitting
2. perform Bayesian fitting

for both the Delta_M model (equation [1] from Chappell paper https://doi.org/10.1002/mrm.22320, and the extended Delta_M_model (equation [1] and [2] from that paper))

In both algorithms, it is possible to flexibly read in the parameters as nparray, as nifti image, or as scalar.
There is the option of using the values of the LS fitting voxel by voxel as prior for the Bayesian fitting.
"""

def ls_fit_voxel(t, signal, m0a, tau):
	"""
	Least squares fitting for a single voxel
	"""

	def model_func(t, att, cbf):
		return deltaM_model(t, att, cbf, m0a, tau)

	param0 = [1.2, 60]
	bounds = ([0.1, 10], [2.5, 200])
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
				m0 = m0_data[x, y, z]

				# Check for invalid signal data
				if (np.any(np.isnan(signal)) or
						np.any(np.isinf(signal)) or
						np.all(signal == 0)):
					continue

				# Check for invalid M0 data
				if (np.isnan(m0) or np.isinf(m0) or m0 <= 0):
					continue

				signal_normalized = signal / m0
				m0a = 1.0

				try:
					att, cbf = ls_fit_voxel(t, signal_normalized, m0a, tau)
					att_map[x, y, z] = att
					cbf_map[x, y, z] = cbf
				except Exception as e:
					continue

	return att_map, cbf_map


def convert_parameter(param_input, shape, param_name="parameter"):
	"""
	Convert parameter input to image / matrix format

	Parameters:
	- param_input: can be scalar, filename (str), or numpy array
	- shape: target shape (x, y, z)
	- param_name: parameter name for error messages

	Returns:
	- 3D numpy array with parameter values
	"""
	if isinstance(param_input, (int, float)):
		# param_input of type float/int will be converted to 3D array
		print(f"Using constant {param_name} = {param_input}")
		return np.full(shape, param_input, dtype=float)

	elif isinstance(param_input, str):
		# param_input is loaded from file
		print(f"Loading {param_name} from file: {param_input}")
		if not os.path.exists(param_input):
			raise FileNotFoundError(f"Parameter file not found: {param_input}")

		_, data = load_nifti_file(param_input)

		# Handle 4D data (take first volume)
		if data.ndim == 4:
			data = data[:, :, :, 0]

		# Check whether shapes are similar
		if data.shape != shape:
			raise ValueError(f"Shape mismatch for {param_name}: expected {shape}, got {data.shape}")

		return data.astype(float)

	elif isinstance(param_input, np.ndarray):
		# Already a matrix
		print(f"Using provided {param_name} array")

		# Handle 4D data (take first volume)
		if param_input.ndim == 4:
			param_input = param_input[:, :, :, 0]

		# Check whether shapes are similar
		if param_input.shape != shape:
			raise ValueError(f"Shape mismatch for {param_name}: expected {shape}, got {param_input.shape}")

		return param_input.astype(float)

	else:
		raise ValueError(f"Invalid {param_name} data format. Must be scalar, filename, or numpy array")


def create_parameter_config():
	"""
	Create extended parameter configuration for deltaM_model_ext
	"""
	return {
		# Fixed physiological parameters
		'T1_blood': 1.65,  # Can be scalar, filename, or nparray
		'T1_tissue': 1.3,  # Can be scalar, filename, or nparray
		'lambda_blood': 0.9,  # Constant scalar
		'alpha': 0.85 * 0.8,  # Constant scalar

		'abv': 0.02,  # Arterial blood volume
		'att_a': 0.7,  # Arterial arrival time (default: 0.7s)
		# Fitting configuration
		'fit_T1_blood': False,  # fit T1_blood as free parameter
		'fit_T1_tissue': False,  # fit T1_tissue as free parameter
		'fit_lambda': False,  # fit lambda as free parameter
		'fit_abv': False,  # fit arterial blood volume
		'fit_att_a': False,  # fit arterial arrival time

		# Priors for free parameters
		'T1_blood_prior': {'mean': 1.65, 'std': 0.2},
		'T1_tissue_prior': {'mean': 1.3, 'std': 0.3},
		'lambda_prior': {'mean': 0.9, 'std': 0.1},
		'abv_prior': {'mean': 0.02, 'std': 0.01},
		'att_a_prior': {'mean': 0.7, 'std': 0.2},

		# ATT and CBF priors from LS fitting
		'att_prior_from_ls': True,  # Use LS ATT results as priors
		'cbf_prior_from_ls': True,  # Use LS CBF results as priors
		'att_prior_std': 0.3,  # Standard deviation for ATT prior
		'cbf_prior_std': 15.0,  # Standard deviation for CBF prior

		# Bounds for free parameters
		'T1_blood_bounds': [1.0, 2.5],
		'T1_tissue_bounds': [0.8, 2.0],
		'lambda_bounds': [0.7, 1.1],
		'abv_bounds': [0.001, 0.1],
		'att_a_bounds': [0.1, 1.5]
	}


STAN_MODEL_CODE = """
functions {
  vector deltaM_model(vector t, real att, real cbf, real m0a, real tau, 
                     real T1a, real T1t, real lambda_val, real alpha) {
    int n = num_elements(t);
    vector[n] deltaM = rep_vector(0.0, n);

    // FIXED: Use corrected deltaM model
    // Common factors
    real cbf_factor = 2 * alpha * cbf / 6000.0;
    real m0_factor = m0a * exp(-att / T1a);
    real t1_lambda_factor = T1a / lambda_val;

    for (i in 1:n) {
      if (t[i] > att && t[i] <= att + tau) {
        // During bolus: att < t <= att + tau
        real exp_term = 1.0 - exp(-(t[i] - att) / T1a);
        deltaM[i] = cbf_factor * m0_factor * t1_lambda_factor * exp_term;
      } else if (t[i] > att + tau) {
        // After bolus: t > att + tau
        real exp1 = exp(-(t[i] - att) / T1a);
        real exp2 = exp(-(t[i] - att - tau) / T1a);
        deltaM[i] = cbf_factor * m0_factor * t1_lambda_factor * (exp1 - exp2);
      }
      // else: t <= att, deltaM[i] remains 0
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

  // Parameter values (used if not fitted)
  real<lower=0> T1a_fixed;
  real<lower=0> T1t_fixed;
  real<lower=0> lambda_fixed;
  real<lower=0> alpha_fixed;

  // Fitting flags
  int<lower=0,upper=1> fit_T1a;
  int<lower=0,upper=1> fit_T1t;
  int<lower=0,upper=1> fit_lambda;

  // Priors (used if fitted)
  real T1a_prior_mean;
  real T1a_prior_std;
  real T1t_prior_mean;
  real T1t_prior_std;
  real lambda_prior_mean;
  real lambda_prior_std;

  // ATT and CBF priors from LS fitting
  int<lower=0,upper=1> use_att_prior_from_ls;
  int<lower=0,upper=1> use_cbf_prior_from_ls;
  real att_prior_from_ls;
  real att_prior_std;
  real cbf_prior_from_ls;
  real cbf_prior_std;

  // Bounds
  real T1a_lower;
  real T1a_upper;
  real T1t_lower;
  real T1t_upper;
  real lambda_lower;
  real lambda_upper;
}

parameters {
  real<lower=0.1, upper=3.0> att;
  real<lower=10, upper=200> cbf;
  real<lower=0.001> sigma;

  // Conditional parameters
  real<lower=T1a_lower, upper=T1a_upper> T1a_param;
  real<lower=T1t_lower, upper=T1t_upper> T1t_param;
  real<lower=lambda_lower, upper=lambda_upper> lambda_param;
}

transformed parameters {
  real T1a_use;
  real T1t_use;
  real lambda_use;

  // Use fitted or fixed values
  T1a_use = fit_T1a ? T1a_param : T1a_fixed;
  T1t_use = fit_T1t ? T1t_param : T1t_fixed;
  lambda_use = fit_lambda ? lambda_param : lambda_fixed;
}

model {
  vector[n] mu;

  // FIXED: Add LS-based priors for ATT and CBF
  if (use_att_prior_from_ls == 1) {
    att ~ normal(att_prior_from_ls, att_prior_std);
  } else {
    att ~ normal(1.2, 0.5);
  }

  if (use_cbf_prior_from_ls == 1) {
    cbf ~ normal(cbf_prior_from_ls, cbf_prior_std);
  } else {
    cbf ~ normal(60, 15);
  }

  sigma ~ exponential(1);

  // Conditional priors for flexible parameters
  if (fit_T1a) {
    T1a_param ~ normal(T1a_prior_mean, T1a_prior_std);
  }
  if (fit_T1t) {
    T1t_param ~ normal(T1t_prior_mean, T1t_prior_std);
  }
  if (fit_lambda) {
    lambda_param ~ normal(lambda_prior_mean, lambda_prior_std);
  }

  // Likelihood
  mu = deltaM_model(t, att, cbf, m0a, tau, T1a_use, T1t_use, lambda_use, alpha_fixed);
  signal ~ normal(mu, sigma);
}
"""

STAN_MODEL_CODE_EXT = """
functions {
  vector deltaM_model_ext(vector t, real att, real cbf, real m0a, real tau, 
                         real abv, real att_a, real T1a, real T1t, real lambda_val, real alpha) {
    int n = num_elements(t);
    vector[n] deltaM = rep_vector(0.0, n);

    // Tissue compartment (Equation [1])
    real cbf_factor = 2 * alpha * cbf / 6000.0;
    real m0_factor = m0a * exp(-att / T1a);
    real t1_lambda_factor = T1a / lambda_val;

    for (i in 1:n) {
      // Tissue contribution
      if (t[i] > att && t[i] <= att + tau) {
        // During bolus: att < t <= att + tau
        real exp_term = 1.0 - exp(-(t[i] - att) / T1a);
        deltaM[i] += cbf_factor * m0_factor * t1_lambda_factor * exp_term;
      } else if (t[i] > att + tau) {
        // After bolus: t > att + tau
        real exp1 = exp(-(t[i] - att) / T1a);
        real exp2 = exp(-(t[i] - att - tau) / T1a);
        deltaM[i] += cbf_factor * m0_factor * t1_lambda_factor * (exp1 - exp2);
      }

      // Arterial contribution (Equation [2])
      if (t[i] >= att_a && t[i] <= att_a + tau) {
        deltaM[i] += (2 * alpha * abv / lambda_val) * m0a * exp(-att_a / T1a);
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

  // Parameter values (used if not fitted)
  real<lower=0> T1a_fixed;
  real<lower=0> T1t_fixed;
  real<lower=0> lambda_fixed;
  real<lower=0> alpha_fixed;
  real<lower=0> abv_fixed;
  real<lower=0> att_a_fixed;

  // Fitting flags
  int<lower=0,upper=1> fit_T1a;
  int<lower=0,upper=1> fit_T1t;
  int<lower=0,upper=1> fit_lambda;
  int<lower=0,upper=1> fit_abv;
  int<lower=0,upper=1> fit_att_a;

  // Priors (used if fitted)
  real T1a_prior_mean;
  real T1a_prior_std;
  real T1t_prior_mean;
  real T1t_prior_std;
  real lambda_prior_mean;
  real lambda_prior_std;
  real abv_prior_mean;
  real abv_prior_std;
  real att_a_prior_mean;
  real att_a_prior_std;

  // ATT and CBF priors from LS fitting
  int<lower=0,upper=1> use_att_prior_from_ls;
  int<lower=0,upper=1> use_cbf_prior_from_ls;
  real att_prior_from_ls;
  real att_prior_std;
  real cbf_prior_from_ls;
  real cbf_prior_std;

  // Bounds
  real T1a_lower;
  real T1a_upper;
  real T1t_lower;
  real T1t_upper;
  real lambda_lower;
  real lambda_upper;
  real abv_lower;
  real abv_upper;
  real att_a_lower;
  real att_a_upper;
}

parameters {
  real<lower=0.1, upper=3.0> att;
  real<lower=10, upper=200> cbf;
  real<lower=0.001> sigma;

  // Conditional parameters
  real<lower=T1a_lower, upper=T1a_upper> T1a_param;
  real<lower=T1t_lower, upper=T1t_upper> T1t_param;
  real<lower=lambda_lower, upper=lambda_upper> lambda_param;
  real<lower=abv_lower, upper=abv_upper> abv_param;
  real<lower=att_a_lower, upper=att_a_upper> att_a_param;
}

transformed parameters {
  real T1a_use;
  real T1t_use;
  real lambda_use;
  real abv_use;
  real att_a_use;

  // Use fitted or constant fixed values
  T1a_use = fit_T1a ? T1a_param : T1a_fixed;
  T1t_use = fit_T1t ? T1t_param : T1t_fixed;
  lambda_use = fit_lambda ? lambda_param : lambda_fixed;
  abv_use = fit_abv ? abv_param : abv_fixed;
  att_a_use = fit_att_a ? att_a_param : att_a_fixed;
}

model {
  vector[n] mu;

  // Priors for ATT and CBF
  if (use_att_prior_from_ls == 1) {
    att ~ normal(att_prior_from_ls, att_prior_std);
  } else {
    att ~ normal(1.2, 0.5);
  }

  if (use_cbf_prior_from_ls == 1) {
    cbf ~ normal(cbf_prior_from_ls, cbf_prior_std);
  } else {
    cbf ~ normal(60, 15);
  }

  sigma ~ exponential(1);

  // Conditional priors for flexible parameters
  if (fit_T1a) {
    T1a_param ~ normal(T1a_prior_mean, T1a_prior_std);
  }
  if (fit_T1t) {
    T1t_param ~ normal(T1t_prior_mean, T1t_prior_std);
  }
  if (fit_lambda) {
    lambda_param ~ normal(lambda_prior_mean, lambda_prior_std);
  }
  if (fit_abv) {
    abv_param ~ normal(abv_prior_mean, abv_prior_std);
  }
  if (fit_att_a) {
    att_a_param ~ normal(att_a_prior_mean, att_a_prior_std);
  }

  // Likelihood
  mu = deltaM_model_ext(t, att, cbf, m0a, tau, abv_use, att_a_use, 
                       T1a_use, T1t_use, lambda_use, alpha_fixed);
  signal ~ normal(mu, sigma);
}
"""


def bayesian_fit_voxel(t, signal, m0a, tau,
					   T1_blood_val, T1_tissue_val, lambda_val, alpha_val,
					   param_config, att_ls_val=None, cbf_ls_val=None):
	"""
	Bayesian fitting for a single voxel with flexible parameters

	Parameters:
	- t: time points
	- signal: signal values (SHOULD BE NORMALIZED BY M0 ALREADY!)
	- m0a: arterial M0 value (should be 1.0 for normalized signals)
	- tau: labeling duration
	- T1_blood_val: T1 blood value for this voxel
	- T1_tissue_val: T1 tissue value for this voxel
	- lambda_val: lambda value for this voxel
	- alpha_val: alpha value
	- param_config: parameter configuration
	- att_ls_val: ATT value from LS fitting (for prior)
	- cbf_ls_val: CBF value from LS fitting (for prior)
	"""

	# Prepare data for using Stan
	data = {
		'n': len(t),
		't': t.astype(float),
		'signal': signal.astype(float),
		'm0a': float(m0a),  # Should be 1.0 for normalized signals
		'tau': float(tau),

		# Fixed parameter values
		'T1a_fixed': float(T1_blood_val),
		'T1t_fixed': float(T1_tissue_val),
		'lambda_fixed': float(lambda_val),
		'alpha_fixed': float(alpha_val),

		# Fitting config
		'fit_T1a': int(param_config['fit_T1_blood']),
		'fit_T1t': int(param_config['fit_T1_tissue']),
		'fit_lambda': int(param_config['fit_lambda']),

		# Priors
		'T1a_prior_mean': float(param_config['T1_blood_prior']['mean']),
		'T1a_prior_std': float(param_config['T1_blood_prior']['std']),
		'T1t_prior_mean': float(param_config['T1_tissue_prior']['mean']),
		'T1t_prior_std': float(param_config['T1_tissue_prior']['std']),
		'lambda_prior_mean': float(param_config['lambda_prior']['mean']),
		'lambda_prior_std': float(param_config['lambda_prior']['std']),

		# LS-based priors
		'use_att_prior_from_ls': int(
			param_config.get('att_prior_from_ls', False) and att_ls_val is not None and np.isfinite(att_ls_val)),
		'use_cbf_prior_from_ls': int(
			param_config.get('cbf_prior_from_ls', False) and cbf_ls_val is not None and np.isfinite(cbf_ls_val)),
		'att_prior_from_ls': float(att_ls_val) if att_ls_val is not None and np.isfinite(att_ls_val) else 1.2,
		'att_prior_std': float(param_config.get('att_prior_std', 0.3)),
		'cbf_prior_from_ls': float(cbf_ls_val) if cbf_ls_val is not None and np.isfinite(cbf_ls_val) else 60.0,
		'cbf_prior_std': float(param_config.get('cbf_prior_std', 15.0)),

		# Bounds
		'T1a_lower': float(param_config['T1_blood_bounds'][0]),
		'T1a_upper': float(param_config['T1_blood_bounds'][1]),
		'T1t_lower': float(param_config['T1_tissue_bounds'][0]),
		'T1t_upper': float(param_config['T1_tissue_bounds'][1]),
		'lambda_lower': float(param_config['lambda_bounds'][0]),
		'lambda_upper': float(param_config['lambda_bounds'][1])
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


def bayesian_fit_volume(pwi_data, t, m0_data, tau, param_config, att_ls_map=None, cbf_ls_map=None):
	"""
	Bayesian fitting for entire volume with flexible parameters

	Parameters:
	- pwi_data: 4D PWI data
	- t: time points
	- m0_data: M0 data
	- tau: labeling duration
	- param_config: parameter configuration dict
	- att_ls_map: ATT map from LS fitting (for priors)
	- cbf_ls_map: CBF map from LS fitting (for priors)
	"""

	shape = pwi_data.shape[:3]

	# Prepare parameter maps
	print("Preparing parameter maps...")
	T1_blood_map = convert_parameter(param_config['T1_blood'], shape, "T1_blood")
	T1_tissue_map = convert_parameter(param_config['T1_tissue'], shape, "T1_tissue")
	lambda_map = convert_parameter(param_config['lambda_blood'], shape, "lambda_blood")
	alpha_map = convert_parameter(param_config['alpha'], shape, "alpha")

	# InitialISe result arrays
	att_map = np.full(shape, np.nan)
	cbf_map = np.full(shape, np.nan)
	att_std_map = np.full(shape, np.nan)
	cbf_std_map = np.full(shape, np.nan)

	# Additional maps for fitted parameters
	if param_config['fit_T1_blood']:
		T1_blood_fitted_map = np.full(shape, np.nan)
		T1_blood_fitted_std_map = np.full(shape, np.nan)

	if param_config['fit_T1_tissue']:
		T1_tissue_fitted_map = np.full(shape, np.nan)
		T1_tissue_fitted_std_map = np.full(shape, np.nan)

	if param_config['fit_lambda']:
		lambda_fitted_map = np.full(shape, np.nan)
		lambda_fitted_std_map = np.full(shape, np.nan)

	total_voxels = shape[0] * shape[1] * shape[2]
	processed_voxels = 0
	successful_fits = 0

	for x in range(shape[0]):
		for y in range(shape[1]):
			for z in range(shape[2]):
				processed_voxels += 1

				if processed_voxels % 100 == 0:
					print(f"Processing voxel {processed_voxels}/{total_voxels} "
						  f"({100 * processed_voxels / total_voxels:.1f}%)")

				# Get data for this voxel
				signal = pwi_data[x, y, z, :]
				m0 = m0_data[x, y, z]

				# Skip invalid voxels
				if (np.any(np.isnan(signal)) or np.any(np.isinf(signal)) or
						np.all(signal == 0) or np.isnan(m0) or np.isinf(m0) or m0 <= 0):
					continue

				# Get parameter values for this voxel
				T1_blood_val = T1_blood_map[x, y, z]
				T1_tissue_val = T1_tissue_map[x, y, z]
				lambda_val = lambda_map[x, y, z]
				alpha_val = alpha_map[x, y, z]

				# Skip if any parameter is invalid
				if (np.isnan(T1_blood_val) or np.isnan(T1_tissue_val) or
						np.isnan(lambda_val) or np.isnan(alpha_val)):
					continue

				# Get LS prior values
				att_ls_val = att_ls_map[x, y, z] if att_ls_map is not None else None
				cbf_ls_val = cbf_ls_map[x, y, z] if cbf_ls_map is not None else None


				signal_normalized = signal / m0
				m0a = 1.0

				try:
					# Fit this voxel
					fit = bayesian_fit_voxel(
						t, signal_normalized, m0a, tau,
						T1_blood_val, T1_tissue_val, lambda_val, alpha_val,
						param_config, att_ls_val, cbf_ls_val
					)

					if fit is not None:
						# Extract results
						att_samples = fit['att'].flatten()
						cbf_samples = fit['cbf'].flatten()

						att_map[x, y, z] = np.mean(att_samples)
						cbf_map[x, y, z] = np.mean(cbf_samples)
						att_std_map[x, y, z] = np.std(att_samples)
						cbf_std_map[x, y, z] = np.std(cbf_samples)

						# Extract fitted parameters
						if param_config['fit_T1_blood']:
							T1_blood_samples = fit['T1a_param'].flatten()
							T1_blood_fitted_map[x, y, z] = np.mean(T1_blood_samples)
							T1_blood_fitted_std_map[x, y, z] = np.std(T1_blood_samples)

						if param_config['fit_T1_tissue']:
							T1_tissue_samples = fit['T1t_param'].flatten()
							T1_tissue_fitted_map[x, y, z] = np.mean(T1_tissue_samples)
							T1_tissue_fitted_std_map[x, y, z] = np.std(T1_tissue_samples)

						if param_config['fit_lambda']:
							lambda_samples = fit['lambda_param'].flatten()
							lambda_fitted_map[x, y, z] = np.mean(lambda_samples)
							lambda_fitted_std_map[x, y, z] = np.std(lambda_samples)

						successful_fits += 1

				except Exception as e:
					print(f"Error fitting voxel ({x}, {y}, {z}): {e}")
					continue

	print(f"Successfully fitted {successful_fits}/{processed_voxels} voxels")

	# Results
	results = {
		'att_map': att_map,
		'cbf_map': cbf_map,
		'att_std_map': att_std_map,
		'cbf_std_map': cbf_std_map,
		'successful_fits': successful_fits,
		'total_processed': processed_voxels
	}

	# Add fitted parameter maps
	if param_config['fit_T1_blood']:
		results['T1_blood_fitted_map'] = T1_blood_fitted_map
		results['T1_blood_fitted_std_map'] = T1_blood_fitted_std_map

	if param_config['fit_T1_tissue']:
		results['T1_tissue_fitted_map'] = T1_tissue_fitted_map
		results['T1_tissue_fitted_std_map'] = T1_tissue_fitted_std_map

	if param_config['fit_lambda']:
		results['lambda_fitted_map'] = lambda_fitted_map
		results['lambda_fitted_std_map'] = lambda_fitted_std_map

	return results


def choose_parameter_config():
	"""
	Choose parameter configuration
	"""

	# Create parameter configuration
	param_config = create_parameter_config()

	# All parameters constant
	param_config['T1_blood'] = 1.65
	param_config['T1_tissue'] = 1.3
	param_config['lambda_blood'] = 0.9

	# Enable LS-based priors
	param_config['att_prior_from_ls'] = True
	param_config['cbf_prior_from_ls'] = True
	param_config['att_prior_std'] = 0.3  # Adjust as needed
	param_config['cbf_prior_std'] = 15.0  # Adjust as needed

	# Load T1 maps from files
	# param_config['T1_blood'] = "path/to/T1_blood_map.nii"
	# param_config['T1_tissue'] = "path/to/T1_tissue_map.nii"

	# Fit T1_blood as free parameter
	# param_config['fit_T1_blood'] = True
	# param_config['T1_blood_prior'] = {'mean': 1.65, 'std': 0.2}

	return param_config


def bayesian_fit_voxel_ext(t, signal, m0a, tau,
						   T1_blood_val, T1_tissue_val, lambda_val, alpha_val,
						   abv_val, att_a_val, param_config,
						   att_ls_val=None, cbf_ls_val=None,
						   abv_ls_val=None, att_a_ls_val=None):
	"""
	Bayesian fitting for a single voxel with extended model and flexible parameters
	"""

	# Prepare data for Stan
	data = {
		'n': len(t),
		't': t.astype(float),
		'signal': signal.astype(float),
		'm0a': float(m0a),
		'tau': float(tau),

		# Fixed parameter values
		'T1a_fixed': float(T1_blood_val),
		'T1t_fixed': float(T1_tissue_val),
		'lambda_fixed': float(lambda_val),
		'alpha_fixed': float(alpha_val),
		'abv_fixed': float(abv_val),
		'att_a_fixed': float(att_a_val),

		# Fitting config
		'fit_T1a': int(param_config['fit_T1_blood']),
		'fit_T1t': int(param_config['fit_T1_tissue']),
		'fit_lambda': int(param_config['fit_lambda']),
		'fit_abv': int(param_config['fit_abv']),
		'fit_att_a': int(param_config['fit_att_a']),

		# Priors - use LS values if available, otherwise use default priors
		'T1a_prior_mean': float(param_config['T1_blood_prior']['mean']),
		'T1a_prior_std': float(param_config['T1_blood_prior']['std']),
		'T1t_prior_mean': float(param_config['T1_tissue_prior']['mean']),
		'T1t_prior_std': float(param_config['T1_tissue_prior']['std']),
		'lambda_prior_mean': float(param_config['lambda_prior']['mean']),
		'lambda_prior_std': float(param_config['lambda_prior']['std']),

		# ABV prior - use LS value if available
		'abv_prior_mean': float(abv_ls_val) if abv_ls_val is not None and np.isfinite(abv_ls_val) else float(
			param_config['abv_prior']['mean']),
		'abv_prior_std': float(param_config.get('abv_prior_std', param_config['abv_prior']['std'])),

		# ATT_A prior - use LS value if available
		'att_a_prior_mean': float(att_a_ls_val) if att_a_ls_val is not None and np.isfinite(att_a_ls_val) else float(
			param_config['att_a_prior']['mean']),
		'att_a_prior_std': float(param_config.get('att_a_prior_std', param_config['att_a_prior']['std'])),

		# LS-based priors for ATT and CBF
		'use_att_prior_from_ls': int(
			param_config.get('att_prior_from_ls', False) and att_ls_val is not None and np.isfinite(att_ls_val)),
		'use_cbf_prior_from_ls': int(
			param_config.get('cbf_prior_from_ls', False) and cbf_ls_val is not None and np.isfinite(cbf_ls_val)),
		'att_prior_from_ls': float(att_ls_val) if att_ls_val is not None and np.isfinite(att_ls_val) else 1.2,
		'att_prior_std': float(param_config.get('att_prior_std', 0.3)),
		'cbf_prior_from_ls': float(cbf_ls_val) if cbf_ls_val is not None and np.isfinite(cbf_ls_val) else 60.0,
		'cbf_prior_std': float(param_config.get('cbf_prior_std', 15.0)),

		# Bounds
		'T1a_lower': float(param_config['T1_blood_bounds'][0]),
		'T1a_upper': float(param_config['T1_blood_bounds'][1]),
		'T1t_lower': float(param_config['T1_tissue_bounds'][0]),
		'T1t_upper': float(param_config['T1_tissue_bounds'][1]),
		'lambda_lower': float(param_config['lambda_bounds'][0]),
		'lambda_upper': float(param_config['lambda_bounds'][1]),
		'abv_lower': float(param_config['abv_bounds'][0]),
		'abv_upper': float(param_config['abv_bounds'][1]),
		'att_a_lower': float(param_config['att_a_bounds'][0]),
		'att_a_upper': float(param_config['att_a_bounds'][1])
	}

	try:
		# Build the model with extended code
		model = stan.build(STAN_MODEL_CODE_EXT, data=data)

		# Sample from the posterior
		fit = model.sample(num_chains=2, num_samples=1000, num_warmup=500)

		return fit

	except Exception as e:
		print(f"Error in Stan fitting: {e}")
		return None



def bayesian_fit_subset(pwi_data, t, m0_data, tau, param_maps, param_config,
						att_map_lm, cbf_map_lm, config):
	"""
	Perform Bayesian fitting on a subset of voxels with flexible parameter handling and LS-based priors.

	Parameters:
	- pwi_data: 4D PWI data
	- t: time points
	- m0_data: M0 data
	- tau: labeling duration
	- param_config: parameter configuration
	- att_map_lm: ATT map from LM fitting (for voxel selection and priors)
	- cbf_map_lm: CBF map from LM fitting (for voxel selection and priors)
	- config: Bayesian fitting configuration
	"""

	# Select voxels based on configuration
	selected_voxels = select_voxels_for_bayesian_fitting(
		att_map_lm, cbf_map_lm, pwi_data, m0_data, param_maps, config
	)

	if len(selected_voxels) == 0:
		print("No suitable voxels found for Bayesian fitting")
		return None

	print(f"Selected {len(selected_voxels)} voxels for Bayesian fitting")

	# Initialise result arrays
	shape = pwi_data.shape[:3]
	att_map = np.full(shape, np.nan)
	cbf_map = np.full(shape, np.nan)
	att_std_map = np.full(shape, np.nan)
	cbf_std_map = np.full(shape, np.nan)

	# Initialise fitted parameter maps if needed
	fitted_param_maps = {}
	if param_config['fit_T1_blood']:
		fitted_param_maps['T1_blood_fitted_map'] = np.full(shape, np.nan)
		fitted_param_maps['T1_blood_fitted_std_map'] = np.full(shape, np.nan)
	if param_config['fit_T1_tissue']:
		fitted_param_maps['T1_tissue_fitted_map'] = np.full(shape, np.nan)
		fitted_param_maps['T1_tissue_fitted_std_map'] = np.full(shape, np.nan)
	if param_config['fit_lambda']:
		fitted_param_maps['lambda_fitted_map'] = np.full(shape, np.nan)
		fitted_param_maps['lambda_fitted_std_map'] = np.full(shape, np.nan)

	# Store results
	individual_results = []
	successful_fits = 0
	failed_fits = 0

	# Process each selected voxel
	for i, (x, y, z) in enumerate(selected_voxels):
		print(f"Processing voxel {i + 1}/{len(selected_voxels)}: ({x}, {y}, {z})")

		# Get data for this voxel
		signal = pwi_data[x, y, z, :]
		m0 = m0_data[x, y, z]

		# Filter out NaN values
		valid_mask = np.isfinite(signal) & (signal != 0)
		signal_clean = signal[valid_mask]
		t_clean = t[valid_mask]

		if m0 > 0 and np.isfinite(m0):
			signal_normalized = signal_clean / m0
		else:
			print(f"  Skipping voxel ({x}, {y}, {z}): invalid M0 value")
			failed_fits += 1
			continue

		# Get parameter values for this voxel
		T1_blood_val = param_maps['T1_blood'][x, y, z]
		T1_tissue_val = param_maps['T1_tissue'][x, y, z]
		lambda_val = param_maps['lambda'][x, y, z]
		alpha_val = param_maps['alpha'][x, y, z]

		# Skip if any parameter is NaN
		if (np.isnan(T1_blood_val) or np.isnan(T1_tissue_val) or
				np.isnan(lambda_val) or np.isnan(alpha_val)):
			print(f"  Skipping voxel ({x}, {y}, {z}): invalid parameter values")
			failed_fits += 1
			continue

		# Get LS prior values for this voxel
		att_ls_val = att_map_lm[x, y, z] if np.isfinite(att_map_lm[x, y, z]) else None
		cbf_ls_val = cbf_map_lm[x, y, z] if np.isfinite(cbf_map_lm[x, y, z]) else None

		if param_config.get('att_prior_from_ls', False) and att_ls_val is not None:
			print(f"  Using LS ATT prior: {att_ls_val:.3f}s")
		if param_config.get('cbf_prior_from_ls', False) and cbf_ls_val is not None:
			print(f"  Using LS CBF prior: {cbf_ls_val:.1f} ml/min/100g")

		# Calculate m0a
		m0a = 1.0

		try:
			# Run Bayesian fitting with flexible parameters and LS priors
			fit = bayesian_fit_voxel(
				t_clean, signal_clean, m0a, tau,
				T1_blood_val, T1_tissue_val, lambda_val, alpha_val,
				param_config, att_ls_val, cbf_ls_val
			)

			if fit is not None:
				# Extract results
				att_samples = fit['att'].flatten()
				cbf_samples = fit['cbf'].flatten()

				att_mean = np.mean(att_samples)
				cbf_mean = np.mean(cbf_samples)
				att_std = np.std(att_samples)
				cbf_std = np.std(cbf_samples)

				# Store in maps
				att_map[x, y, z] = att_mean
				cbf_map[x, y, z] = cbf_mean
				att_std_map[x, y, z] = att_std
				cbf_std_map[x, y, z] = cbf_std

				# Store fitted parameter results
				fitted_results = {}
				if param_config['fit_T1_blood']:
					T1_blood_samples = fit['T1a_param'].flatten()
					T1_blood_fitted_mean = np.mean(T1_blood_samples)
					T1_blood_fitted_std = np.std(T1_blood_samples)
					fitted_param_maps['T1_blood_fitted_map'][x, y, z] = T1_blood_fitted_mean
					fitted_param_maps['T1_blood_fitted_std_map'][x, y, z] = T1_blood_fitted_std
					fitted_results['T1_blood_fitted_mean'] = T1_blood_fitted_mean
					fitted_results['T1_blood_fitted_std'] = T1_blood_fitted_std

				if param_config['fit_T1_tissue']:
					T1_tissue_samples = fit['T1t_param'].flatten()
					T1_tissue_fitted_mean = np.mean(T1_tissue_samples)
					T1_tissue_fitted_std = np.std(T1_tissue_samples)
					fitted_param_maps['T1_tissue_fitted_map'][x, y, z] = T1_tissue_fitted_mean
					fitted_param_maps['T1_tissue_fitted_std_map'][x, y, z] = T1_tissue_fitted_std
					fitted_results['T1_tissue_fitted_mean'] = T1_tissue_fitted_mean
					fitted_results['T1_tissue_fitted_std'] = T1_tissue_fitted_std

				if param_config['fit_lambda']:
					lambda_samples = fit['lambda_param'].flatten()
					lambda_fitted_mean = np.mean(lambda_samples)
					lambda_fitted_std = np.std(lambda_samples)
					fitted_param_maps['lambda_fitted_map'][x, y, z] = lambda_fitted_mean
					fitted_param_maps['lambda_fitted_std_map'][x, y, z] = lambda_fitted_std
					fitted_results['lambda_fitted_mean'] = lambda_fitted_mean
					fitted_results['lambda_fitted_std'] = lambda_fitted_std

				# Store results for individual voxel
				individual_result = {
					'voxel': (x, y, z),
					'att_mean': att_mean,
					'att_std': att_std,
					'cbf_mean': cbf_mean,
					'cbf_std': cbf_std,
					'att_lm': att_ls_val if att_ls_val is not None else np.nan,
					'cbf_lm': cbf_ls_val if cbf_ls_val is not None else np.nan,
					'parameters': {
						'T1_blood': T1_blood_val,
						'T1_tissue': T1_tissue_val,
						'lambda': lambda_val,
						'alpha': alpha_val
					},
					'fitted_parameters': fitted_results
				}
				individual_results.append(individual_result)

				successful_fits += 1
				print(f"  Success: ATT={att_mean:.3f}±{att_std:.3f}s, "
					  f"CBF={cbf_mean:.1f}±{cbf_std:.1f} ml/min/100g")

				# Show comparison with LS if available
				if att_ls_val is not None and cbf_ls_val is not None:
					att_diff = att_mean - att_ls_val
					cbf_diff = cbf_mean - cbf_ls_val
					print(f"  from LS: ATT={att_diff:+.3f}s, CBF={cbf_diff:+.1f} ml/min/100g")
			else:
				print(f"  Bayesian fitting failed for voxel ({x}, {y}, {z})")
				failed_fits += 1

		except Exception as e:
			print(f"  Error fitting voxel ({x}, {y}, {z}): {e}")
			failed_fits += 1
			continue

	print(f"\n=== Bayesian Fitting Summary ===")
	print(f"Total selected voxels: {len(selected_voxels)}")
	print(f"Successfully fitted: {successful_fits}/{len(selected_voxels)} "
		  f"({100 * successful_fits / len(selected_voxels):.1f}%)")
	print(f"Failed fits: {failed_fits}/{len(selected_voxels)} "
		  f"({100 * failed_fits / len(selected_voxels):.1f}%)")

	# Results
	results = {
		'att_map': att_map,
		'cbf_map': cbf_map,
		'att_std_map': att_std_map,
		'cbf_std_map': cbf_std_map,
		'individual_results': individual_results,
		'selected_voxels': selected_voxels,
		'successful_fits': successful_fits,
		'failed_fits': failed_fits,
		'total_selected_voxels': len(selected_voxels)
	}

	# Add fitted parameter maps
	results.update(fitted_param_maps)

	return results

def select_voxels_for_bayesian_fitting(att_map_lm, cbf_map_lm, pwi_data, m0_data, param_maps, config):
	"""
	Select voxels for Bayesian fitting
	"""

	valid_mask = (~np.isnan(att_map_lm) & ~np.isnan(cbf_map_lm))

	# Check parameter validity
	for param_name, param_map in param_maps.items():
		valid_mask &= (~np.isnan(param_map) & ~np.isinf(param_map))

	initial_valid_count = np.sum(valid_mask)
	print(f"Valid voxels (non-NaN parameters): {initial_valid_count}")

	shape = pwi_data.shape[:3]
	for x in range(shape[0]):
		for y in range(shape[1]):
			for z in range(shape[2]):
				if valid_mask[x, y, z]:
					signal = pwi_data[x, y, z, :]
					m0 = m0_data[x, y, z]

					# Check for sufficient valid data points and valid M0
					valid_signal_mask = np.isfinite(signal) & (signal != 0)
					if np.sum(valid_signal_mask) < 2 or np.isnan(m0) or np.isinf(m0) or m0 <= 0:
						valid_mask[x, y, z] = False

	# Find all valid coordinates
	valid_coords = np.where(valid_mask)
	all_valid_voxels = list(zip(valid_coords[0], valid_coords[1], valid_coords[2]))

	print(f"Final valid voxels for Bayesian fitting: {len(all_valid_voxels)}")

	if len(all_valid_voxels) == 0:
		return []

	# Select subset
	max_voxels = config['max_voxels']
	selection_method = config['selection_method']

	if selection_method == 'all':
		selected_voxels = all_valid_voxels
		print(f"Using all {len(selected_voxels)} valid voxels")

	elif selection_method == 'random':
		if max_voxels is None or len(all_valid_voxels) <= max_voxels:
			selected_voxels = all_valid_voxels
		else:
			np.random.seed(42)
			selected_indices = np.random.choice(len(all_valid_voxels), max_voxels, replace=False)
			selected_voxels = [all_valid_voxels[i] for i in selected_indices]
			print(f"Randomly selected {len(selected_voxels)} out of {len(all_valid_voxels)} valid voxels")

	elif selection_method == 'grid':
		spacing = config.get('grid_spacing', 5)
		selected_voxels = []
		for x, y, z in all_valid_voxels:
			if (x % spacing == 0 and y % spacing == 0 and z % spacing == 0):
				selected_voxels.append((x, y, z))
				if max_voxels is not None and len(selected_voxels) >= max_voxels:
					break
		print(
			f"Grid selection (spacing={spacing}): {len(selected_voxels)} out of {len(all_valid_voxels)} valid voxels")

	elif selection_method == 'best_lm':
		valid_att = att_map_lm[valid_coords]
		valid_cbf = cbf_map_lm[valid_coords]
		median_att = np.median(valid_att)
		median_cbf = np.median(valid_cbf)
		distances = np.sqrt((valid_att - median_att) ** 2 + (valid_cbf - median_cbf) ** 2)
		sorted_indices = np.argsort(distances)
		if max_voxels is None:
			selected_voxels = [all_valid_voxels[i] for i in sorted_indices]
		else:
			selected_voxels = [all_valid_voxels[i] for i in sorted_indices[:max_voxels]]
		print(f"Best LM selection: {len(selected_voxels)} out of {len(all_valid_voxels)} valid voxels")

	else:
		selected_voxels = all_valid_voxels
		print(f"Using all {len(selected_voxels)} valid voxels")

	return selected_voxels


def bayesian_fit_subset(pwi_data, t, m0_data, tau, param_maps, param_config,
						att_map_lm, cbf_map_lm, config):
	"""
	Perform Bayesian fitting on a subset of voxels and LS-based priors

	Parameters:
	- pwi_data: 4D PWI data
	- t: time points
	- m0_data: M0 data
	- tau: labeling duration
	- param_config: parameter configuration
	- att_map_lm: ATT map from LM fitting (for voxel selection and priors)
	- cbf_map_lm: CBF map from LM fitting (for voxel selection and priors)
	- config: Bayesian fitting configuration
	"""

	# Select voxels based on configuration
	selected_voxels = select_voxels_for_bayesian_fitting(
		att_map_lm, cbf_map_lm, pwi_data, m0_data, param_maps, config
	)

	if len(selected_voxels) == 0:
		print("No suitable voxels found for Bayesian fitting")
		return None

	print(f"Selected {len(selected_voxels)} voxels for Bayesian fitting")

	# Initialise result arrays
	shape = pwi_data.shape[:3]
	att_map = np.full(shape, np.nan)
	cbf_map = np.full(shape, np.nan)
	att_std_map = np.full(shape, np.nan)
	cbf_std_map = np.full(shape, np.nan)

	# Initialise fitted parameter maps if needed
	fitted_param_maps = {}
	if param_config['fit_T1_blood']:
		fitted_param_maps['T1_blood_fitted_map'] = np.full(shape, np.nan)
		fitted_param_maps['T1_blood_fitted_std_map'] = np.full(shape, np.nan)
	if param_config['fit_T1_tissue']:
		fitted_param_maps['T1_tissue_fitted_map'] = np.full(shape, np.nan)
		fitted_param_maps['T1_tissue_fitted_std_map'] = np.full(shape, np.nan)
	if param_config['fit_lambda']:
		fitted_param_maps['lambda_fitted_map'] = np.full(shape, np.nan)
		fitted_param_maps['lambda_fitted_std_map'] = np.full(shape, np.nan)

	# Store results
	individual_results = []
	successful_fits = 0
	failed_fits = 0

	# Process each selected voxel
	for i, (x, y, z) in enumerate(selected_voxels):
		print(f"Processing voxel {i + 1}/{len(selected_voxels)}: ({x}, {y}, {z})")

		# Get data for this voxel
		signal = pwi_data[x, y, z, :]
		m0 = m0_data[x, y, z]

		# Filter out NaN values
		valid_mask = np.isfinite(signal) & (signal != 0)
		signal_clean = signal[valid_mask]
		t_clean = t[valid_mask]

		if m0 > 0 and np.isfinite(m0):
			signal_normalized = signal_clean / m0
		else:
			print(f"  Skipping voxel ({x}, {y}, {z}): invalid M0 value")
			failed_fits += 1
			continue


		# Get parameter values for this voxel
		T1_blood_val = param_maps['T1_blood'][x, y, z]
		T1_tissue_val = param_maps['T1_tissue'][x, y, z]
		lambda_val = param_maps['lambda'][x, y, z]
		alpha_val = param_maps['alpha'][x, y, z]

		# Skip if any parameter is NaN
		if (np.isnan(T1_blood_val) or np.isnan(T1_tissue_val) or
				np.isnan(lambda_val) or np.isnan(alpha_val)):
			print(f"  Skipping voxel ({x}, {y}, {z}): invalid parameter values")
			failed_fits += 1
			continue

		# Get LS prior values for this voxel
		att_ls_val = att_map_lm[x, y, z] if np.isfinite(att_map_lm[x, y, z]) else None
		cbf_ls_val = cbf_map_lm[x, y, z] if np.isfinite(cbf_map_lm[x, y, z]) else None

		if param_config.get('att_prior_from_ls', False) and att_ls_val is not None:
			print(f"  Using LS ATT prior: {att_ls_val:.3f}s")
		if param_config.get('cbf_prior_from_ls', False) and cbf_ls_val is not None:
			print(f"  Using LS CBF prior: {cbf_ls_val:.1f} ml/min/100g")

		# Calculate m0a
		m0a = 1.0

		try:
			# Run Bayesian fitting with flexible parameters and LS priors
			fit = bayesian_fit_voxel(
				t_clean, signal_clean, m0a, tau,
				T1_blood_val, T1_tissue_val, lambda_val, alpha_val,
				param_config, att_ls_val, cbf_ls_val
			)

			if fit is not None:
				# Extract results
				att_samples = fit['att'].flatten()
				cbf_samples = fit['cbf'].flatten()

				att_mean = np.mean(att_samples)
				cbf_mean = np.mean(cbf_samples)
				att_std = np.std(att_samples)
				cbf_std = np.std(cbf_samples)

				# Store in maps
				att_map[x, y, z] = att_mean
				cbf_map[x, y, z] = cbf_mean
				att_std_map[x, y, z] = att_std
				cbf_std_map[x, y, z] = cbf_std

				# Store fitted parameter results
				fitted_results = {}
				if param_config['fit_T1_blood']:
					T1_blood_samples = fit['T1a_param'].flatten()
					T1_blood_fitted_mean = np.mean(T1_blood_samples)
					T1_blood_fitted_std = np.std(T1_blood_samples)
					fitted_param_maps['T1_blood_fitted_map'][x, y, z] = T1_blood_fitted_mean
					fitted_param_maps['T1_blood_fitted_std_map'][x, y, z] = T1_blood_fitted_std
					fitted_results['T1_blood_fitted_mean'] = T1_blood_fitted_mean
					fitted_results['T1_blood_fitted_std'] = T1_blood_fitted_std

				if param_config['fit_T1_tissue']:
					T1_tissue_samples = fit['T1t_param'].flatten()
					T1_tissue_fitted_mean = np.mean(T1_tissue_samples)
					T1_tissue_fitted_std = np.std(T1_tissue_samples)
					fitted_param_maps['T1_tissue_fitted_map'][x, y, z] = T1_tissue_fitted_mean
					fitted_param_maps['T1_tissue_fitted_std_map'][x, y, z] = T1_tissue_fitted_std
					fitted_results['T1_tissue_fitted_mean'] = T1_tissue_fitted_mean
					fitted_results['T1_tissue_fitted_std'] = T1_tissue_fitted_std

				if param_config['fit_lambda']:
					lambda_samples = fit['lambda_param'].flatten()
					lambda_fitted_mean = np.mean(lambda_samples)
					lambda_fitted_std = np.std(lambda_samples)
					fitted_param_maps['lambda_fitted_map'][x, y, z] = lambda_fitted_mean
					fitted_param_maps['lambda_fitted_std_map'][x, y, z] = lambda_fitted_std
					fitted_results['lambda_fitted_mean'] = lambda_fitted_mean
					fitted_results['lambda_fitted_std'] = lambda_fitted_std

				# Store results for individual voxel
				individual_result = {
					'voxel': (x, y, z),
					'att_mean': att_mean,
					'att_std': att_std,
					'cbf_mean': cbf_mean,
					'cbf_std': cbf_std,
					'att_lm': att_ls_val if att_ls_val is not None else np.nan,
					'cbf_lm': cbf_ls_val if cbf_ls_val is not None else np.nan,
					'parameters': {
						'T1_blood': T1_blood_val,
						'T1_tissue': T1_tissue_val,
						'lambda': lambda_val,
						'alpha': alpha_val
					},
					'fitted_parameters': fitted_results
				}
				individual_results.append(individual_result)

				successful_fits += 1
				print(f"  Success: ATT={att_mean:.3f}±{att_std:.3f}s, "
					  f"CBF={cbf_mean:.1f}±{cbf_std:.1f} ml/min/100g")

				# Show comparison with LS if available
				if att_ls_val is not None and cbf_ls_val is not None:
					att_diff = att_mean - att_ls_val
					cbf_diff = cbf_mean - cbf_ls_val
					print(f"  from LS: ATT={att_diff:+.3f}s, CBF={cbf_diff:+.1f} ml/min/100g")
			else:
				print(f"  Bayesian fitting failed for voxel ({x}, {y}, {z})")
				failed_fits += 1

		except Exception as e:
			print(f"  Error fitting voxel ({x}, {y}, {z}): {e}")
			failed_fits += 1
			continue

	print(f"\n=== Bayesian Fitting Summary ===")
	print(f"Total selected voxels: {len(selected_voxels)}")
	print(f"Successfully fitted: {successful_fits}/{len(selected_voxels)} "
		  f"({100 * successful_fits / len(selected_voxels):.1f}%)")
	print(f"Failed fits: {failed_fits}/{len(selected_voxels)} "
		  f"({100 * failed_fits / len(selected_voxels):.1f}%)")

	# Results
	results = {
		'att_map': att_map,
		'cbf_map': cbf_map,
		'att_std_map': att_std_map,
		'cbf_std_map': cbf_std_map,
		'individual_results': individual_results,
		'selected_voxels': selected_voxels,
		'successful_fits': successful_fits,
		'failed_fits': failed_fits,
		'total_selected_voxels': len(selected_voxels)
	}

	# Add fitted parameter maps
	results.update(fitted_param_maps)

	return results


def select_voxels_for_bayesian_fitting(att_map_lm, cbf_map_lm, pwi_data, m0_data, param_maps, config):
	"""
	Select voxels for Bayesian fitting
	"""

	valid_mask = (~np.isnan(att_map_lm) & ~np.isnan(cbf_map_lm))

	# Check parameter validity
	for param_name, param_map in param_maps.items():
		valid_mask &= (~np.isnan(param_map) & ~np.isinf(param_map))

	initial_valid_count = np.sum(valid_mask)
	print(f"Valid voxels (non-NaN parameters): {initial_valid_count}")

	shape = pwi_data.shape[:3]
	for x in range(shape[0]):
		for y in range(shape[1]):
			for z in range(shape[2]):
				if valid_mask[x, y, z]:
					signal = pwi_data[x, y, z, :]
					m0 = m0_data[x, y, z]

					# Check for sufficient valid data points and valid M0
					valid_signal_mask = np.isfinite(signal) & (signal != 0)
					if np.sum(valid_signal_mask) < 2 or np.isnan(m0) or np.isinf(m0) or m0 <= 0:
						valid_mask[x, y, z] = False

	# Find all valid coordinates
	valid_coords = np.where(valid_mask)
	all_valid_voxels = list(zip(valid_coords[0], valid_coords[1], valid_coords[2]))

	print(f"Final valid voxels for Bayesian fitting: {len(all_valid_voxels)}")

	if len(all_valid_voxels) == 0:
		return []

	# Select subset
	max_voxels = config['max_voxels']
	selection_method = config['selection_method']

	if selection_method == 'all':
		selected_voxels = all_valid_voxels
		print(f"Using all {len(selected_voxels)} valid voxels")

	elif selection_method == 'random':
		if max_voxels is None or len(all_valid_voxels) <= max_voxels:
			selected_voxels = all_valid_voxels
		else:
			np.random.seed(42)
			selected_indices = np.random.choice(len(all_valid_voxels), max_voxels, replace=False)
			selected_voxels = [all_valid_voxels[i] for i in selected_indices]
			print(f"Randomly selected {len(selected_voxels)} out of {len(all_valid_voxels)} valid voxels")

	elif selection_method == 'grid':
		spacing = config.get('grid_spacing', 5)
		selected_voxels = []
		for x, y, z in all_valid_voxels:
			if (x % spacing == 0 and y % spacing == 0 and z % spacing == 0):
				selected_voxels.append((x, y, z))
				if max_voxels is not None and len(selected_voxels) >= max_voxels:
					break
		print(f"Grid selection (spacing={spacing}): {len(selected_voxels)} out of {len(all_valid_voxels)} valid voxels")

	elif selection_method == 'best_lm':
		valid_att = att_map_lm[valid_coords]
		valid_cbf = cbf_map_lm[valid_coords]
		median_att = np.median(valid_att)
		median_cbf = np.median(valid_cbf)
		distances = np.sqrt((valid_att - median_att) ** 2 + (valid_cbf - median_cbf) ** 2)
		sorted_indices = np.argsort(distances)
		if max_voxels is None:
			selected_voxels = [all_valid_voxels[i] for i in sorted_indices]
		else:
			selected_voxels = [all_valid_voxels[i] for i in sorted_indices[:max_voxels]]
		print(f"Best LM selection: {len(selected_voxels)} out of {len(all_valid_voxels)} valid voxels")

	else:
		selected_voxels = all_valid_voxels
		print(f"Using all {len(selected_voxels)} valid voxels")

	return selected_voxels



def print_bayesian_summary_flexible(results):
	"""
	Print summary of Bayesian fitting results
	"""
	individual_results = results['individual_results']

	if len(individual_results) == 0:
		print("No successful Bayesian fits to summarize")
		return

	# Extract values
	att_means = [r['att_mean'] for r in individual_results]
	att_stds = [r['att_std'] for r in individual_results]
	cbf_means = [r['cbf_mean'] for r in individual_results]
	cbf_stds = [r['cbf_std'] for r in individual_results]

	print(f"\n=== Bayesian Fitting Results ({len(individual_results)} voxels) ===")
	print(f"ATT (mean): {np.mean(att_means):.3f} ± {np.std(att_means):.3f}s")
	print(f"ATT (range): [{np.min(att_means):.3f}, {np.max(att_means):.3f}]s")
	print(f"ATT uncertainty (mean): {np.mean(att_stds):.3f}s")

	print(f"CBF (mean): {np.mean(cbf_means):.1f} ± {np.std(cbf_means):.1f} ml/min/100g")
	print(f"CBF (range): [{np.min(cbf_means):.1f}, {np.max(cbf_means):.1f}] ml/min/100g")
	print(f"CBF uncertainty (mean): {np.mean(cbf_stds):.1f} ml/min/100g")


def print_comparison_summary(bayesian_results, att_map_lm, cbf_map_lm):
	"""
	Print comparison between Bayesian and LS results
	"""
	individual_results = bayesian_results['individual_results']

	if len(individual_results) == 0:
		print("No results to compare")
		return

	# Extract comparison data
	att_bayes = []
	cbf_bayes = []
	att_ls = []
	cbf_ls = []
	att_diffs = []
	cbf_diffs = []

	for result in individual_results:
		att_bayes.append(result['att_mean'])
		cbf_bayes.append(result['cbf_mean'])

		att_lm_val = result['att_lm']
		cbf_lm_val = result['cbf_lm']

		if np.isfinite(att_lm_val) and np.isfinite(cbf_lm_val):
			att_ls.append(att_lm_val)
			cbf_ls.append(cbf_lm_val)
			att_diffs.append(result['att_mean'] - att_lm_val)
			cbf_diffs.append(result['cbf_mean'] - cbf_lm_val)

	if len(att_diffs) > 0:
		print(f"\n=== Bayesian vs LS Comparison ({len(att_diffs)} voxels) ===")

		# Correlation
		if len(att_ls) > 1:
			att_corr = np.corrcoef(att_bayes[:len(att_ls)], att_ls)[0, 1]
			cbf_corr = np.corrcoef(cbf_bayes[:len(cbf_ls)], cbf_ls)[0, 1]
			print(f"Correlation - ATT: {att_corr:.3f}, CBF: {cbf_corr:.3f}")

		# Mean differences
		print(f"Mean differences (Bayesian - LS):")
		print(f"  ATT: {np.mean(att_diffs):+.3f} ± {np.std(att_diffs):.3f}s")
		print(f"  CBF: {np.mean(cbf_diffs):+.1f} ± {np.std(cbf_diffs):.1f} ml/min/100g")

		# Absolute differences
		print(f"Mean absolute differences:")
		print(f"  ATT: {np.mean(np.abs(att_diffs)):.3f}s")
		print(f"  CBF: {np.mean(np.abs(cbf_diffs)):.1f} ml/min/100g")

		# Range of differences
		print(f"Difference ranges:")
		print(f"  ATT: [{np.min(att_diffs):+.3f}, {np.max(att_diffs):+.3f}]s")
		print(f"  CBF: [{np.min(cbf_diffs):+.1f}, {np.max(cbf_diffs):+.1f}] ml/min/100g")

