import os
import numpy as np
from data_handling import load_nifti_file, load_json_metadata, save_nifti
from fitting_single_te import create_parameter_config_from_config, convert_parameter, ls_fit_volume, ls_fit_volume_ext, \
	bayesian_fit_volume, bayesian_fit_volume_ext
from csv_utils import save_results_to_csv, save_ls_results_summary_csv
from debug_asl import bayesian_fit_subset, print_comparison_summary, print_bayesian_summary_flexible
from config import config

"""
This Python file provides an asl script, which consists of the following steps:

1. Load configuration (parameters are set in a config file)
2. Read and print data
3. Analyse data for NaN values, they are ignored during fitting
4. Perform Least Squares fitting and save results
5. Perform Bayesian fitting, either for the complete volume or for an adjustable subset of voxels, and save results

The fitting can be run for either the tissue compartment only from DeltaM_model (simple model)
or for the extended version which also incorporates the arterial component (extended model)

Output:
- nifti images of the fitted CBF and ATT data
- print values in csv file (optional)
"""


def create_parameter_config_from_config():
	"""
	Create parameter configuration from config.py
	"""
	param_config = {
		# Fixed physiological parameters
		'T1a': config['physiological']['T1a'],
		'T1': config['physiological']['T1'],
		'lambd': config['physiological']['lambd'],
		'a': config['physiological']['a'],
		'abv': config['physiological']['abv'],
		'att_a': config['physiological']['att_a'],

		# Fitting configuration
		'fit_T1a': config['fitting']['fit_T1a'],
		'fit_T1': config['fitting']['fit_T1'],
		'fit_lambd': config['fitting']['fit_lambd'],
		'fit_abv': config['fitting']['fit_abv'],
		'fit_att_a': config['fitting']['fit_att_a'],

		# Priors for free parameters
		'T1a_prior': config['priors']['T1a'],
		'T1_prior': config['priors']['T1'],
		'lambd_prior': config['priors']['lambd'],
		'abv_prior': config['priors']['abv'],
		'att_a_prior': config['priors']['att_a'],

		# ATT and CBF priors from LS fitting
		'att_prior_from_ls': config['ls_priors']['att_prior_from_ls'],
		'cbf_prior_from_ls': config['ls_priors']['cbf_prior_from_ls'],
		'att_prior_std': config['ls_priors']['att_prior_std'],
		'cbf_prior_std': config['ls_priors']['cbf_prior_std'],

		# Bounds for free parameters
		'T1a_bounds': config['bounds']['T1a'],
		'T1_bounds': config['bounds']['T1'],
		'lambd_bounds': config['bounds']['lambd'],
		'abv_bounds': config['bounds']['abv'],
		'att_a_bounds': config['bounds']['att_a']
	}
	return param_config


def create_bayesian_config_from_config():
	"""
	Create Bayesian configuration from config.py
	"""
	return config['bayesian']


def create_ls_config_from_config():
	"""
	Create least squares configuration from config.py
	"""
	return config['least_squares']


def print_ls_summary(att_map_lm, cbf_map_lm, additional_maps=None):
	"""
	Print summary of Least Squares fitting results

	Parameters:
	- att_map_lm: ATT map from LS fitting
	- cbf_map_lm: CBF map from LS fitting
	- additional_maps: dict with additional maps (e.g., {'abv': abv_map, 'att_a': att_a_map})
	"""
	att_finite = att_map_lm[np.isfinite(att_map_lm)]
	cbf_finite = cbf_map_lm[np.isfinite(cbf_map_lm)]

	if len(att_finite) > 0:
		print(f"\n=== Least Squares Results Summary ===")
		print(f"ATT (LS): {np.mean(att_finite):.3f} ± {np.std(att_finite):.3f}s")
		print(f"ATT range: [{np.min(att_finite):.3f}, {np.max(att_finite):.3f}]s")
		print(f"Valid ATT voxels: {len(att_finite)}")

	if len(cbf_finite) > 0:
		print(f"CBF (LS): {np.mean(cbf_finite):.1f} ± {np.std(cbf_finite):.1f} ml/min/100g")
		print(f"CBF range: [{np.min(cbf_finite):.1f}, {np.max(cbf_finite):.1f}] ml/min/100g")
		print(f"Valid CBF voxels: {len(cbf_finite)}")

	# Print additional maps if provided (for extended model)
	if additional_maps:
		for param_name, param_map in additional_maps.items():
			if param_map is not None:
				param_finite = param_map[np.isfinite(param_map)]
				if len(param_finite) > 0:
					print(f"{param_name.upper()} (LS): {np.mean(param_finite):.4f} ± {np.std(param_finite):.4f}")
					print(f"{param_name.upper()} range: [{np.min(param_finite):.4f}, {np.max(param_finite):.4f}]")


def asl(fitting_method='simple'):
	"""
	Main ASL processing function

	Parameters:
	- fitting_method: str, either 'simple' (default) or 'extended'
		- 'simple': uses simple tissue model (dm_tiss only)
		- 'extended': uses extended model with tissue + arterial compartments (dm_tiss + dm_art)
	"""

	# Validate fitting method
	if fitting_method not in ['simple', 'extended']:
		raise ValueError("fitting_method must be either 'simple' or 'extended'")

	print(f"=== ASL Processing using {fitting_method} model ===")

	# 1. === Load configuration ===
	print("=== Loading Configuration ===")
	param_config = create_parameter_config_from_config()
	bayesian_config = create_bayesian_config_from_config()
	ls_config = create_ls_config_from_config()

	# 2. === read and print data ===
	# Data paths
	script_dir = os.path.dirname(os.path.abspath(__file__))
	data_dir = os.path.join(script_dir, "..", "data", "1TE")

	pwi_path = os.path.join(data_dir, "PWI4D.nii")
	pwi_json_path = os.path.join(data_dir, "PWI4D.json")
	m0_path = os.path.join(data_dir, "M0.nii")
	m0_json_path = os.path.join(data_dir, "M0.json")

	# Load data
	print("Loading PWI and M0 data...")
	pwi_img, pwi_data_full = load_nifti_file(pwi_path)
	m0_img, m0_data_full = load_nifti_file(m0_path)
	pwi_meta = load_json_metadata(pwi_json_path)
	m0_meta = load_json_metadata(m0_json_path)

	echo_times = np.array(pwi_meta["EchoTime"])
	plds = np.array(pwi_meta["PostLabelingDelay"])
	tau = pwi_meta.get("LabelingDuration")
	if isinstance(tau, list):
		tau = tau[0]

	# Handle M0 data dimensions
	if m0_data_full.ndim == 4 and m0_data_full.shape[3] == 1:
		m0_data = m0_data_full[:, :, :, 0]
	else:
		m0_data = m0_data_full

	echo_time = echo_times[0]  # since we use single-te data here, we choose the first value
	indices = [i for i, te in enumerate(echo_times) if np.isclose(te, echo_time)]
	pwi_data = pwi_data_full[..., indices]
	t = plds[indices]

	sorted_indices = np.argsort(plds)
	t = plds[sorted_indices]
	pwi_data = pwi_data_full[..., sorted_indices]

	print(f"Using TE = {echo_time} s, tau = {tau} s")
	print(f"PWI shape: {pwi_data.shape}, M0 shape: {m0_data.shape}")

	# 3. === analyse data for NaN values ===
	nan_count = np.sum(np.isnan(pwi_data))
	total_elements = np.prod(pwi_data.shape)
	print(f"NaN values in PWI data: {nan_count}/{total_elements} ({100 * nan_count / total_elements:.2f}%)")

	m0_nan_count = np.sum(np.isnan(m0_data))
	m0_total_elements = np.prod(m0_data.shape)
	print(f"NaN values in M0 data: {m0_nan_count}/{m0_total_elements} ({100 * m0_nan_count / m0_total_elements:.2f}%)")

	# Check data range
	pwi_finite = pwi_data[np.isfinite(pwi_data)]
	m0_finite = m0_data[np.isfinite(m0_data)]

	if len(pwi_finite) > 0:
		print(f"PWI data range: [{np.min(pwi_finite):.6f}, {np.max(pwi_finite):.6f}]")
	if len(m0_finite) > 0:
		print(f"M0 data range: [{np.min(m0_finite):.6f}, {np.max(m0_finite):.6f}]")

	# 4. === Parameter Configuration ===
	print("\n=== Parameter Configuration ===")
	print(f"Using configuration from config.py")

	# Print key parameters
	print(f"T1a = {param_config['T1a']}")
	print(f"T1 = {param_config['T1']}")
	print(f"lambda = {param_config['lambd']}")
	print(f"alpha = {param_config['a']}")

	print(f"Using LS-based priors: ATT={param_config['att_prior_from_ls']}, CBF={param_config['cbf_prior_from_ls']}")
	print(f"Prior standard deviations: ATT={param_config['att_prior_std']}, CBF={param_config['cbf_prior_std']}")

	# Set parameter maps using the correct parameter names
	shape = pwi_data.shape[:3]
	print("Set parameter maps...")
	T1a_map = convert_parameter(param_config['T1a'], shape, "T1a")
	T1_map = convert_parameter(param_config['T1'], shape, "T1")
	lambd_map = convert_parameter(param_config['lambd'], shape, "lambd")
	a_map = convert_parameter(param_config['a'], shape, "a")

	param_maps = {
		'T1a': T1a_map,
		'T1': T1_map,
		'lambd': lambd_map,
		'a': a_map
	}

	# 4. === Least Squares fitting ===
	if fitting_method == 'simple':
		print("\n=== Running Least Squares fitting (simple model) ===")
		att_map_lm, cbf_map_lm = ls_fit_volume(
			pwi_data, t, m0_data, tau,
			param_config['lambd'], param_config['T1'], param_config['T1a'], param_config['a']
		)

		print("Saving LS results (simple model)...")
		save_nifti(att_map_lm, pwi_img, os.path.join(data_dir, "ATT_map_LS.nii.gz"))
		save_nifti(cbf_map_lm, pwi_img, os.path.join(data_dir, "CBF_map_LS.nii.gz"))

		# Print LS results summary
		print_ls_summary(att_map_lm, cbf_map_lm)

		# Initialize additional maps as None for simple model
		abv_map_lm = None
		att_a_map_lm = None
		rms_map = None

	elif fitting_method == 'extended':
		print("\n=== Running Least Squares fitting (Extended Model) ===")
		att_map_lm, cbf_map_lm, abv_map_lm, att_a_map_lm = ls_fit_volume_ext(
			pwi_data, t, m0_data, tau,
			param_config['lambd'], param_config['T1'], param_config['T1a'], param_config['a']
		)

		print("Saving LS results (Extended Model)...")
		save_nifti(att_map_lm, pwi_img, os.path.join(data_dir, "ATT_map_LS_ext.nii.gz"))
		save_nifti(cbf_map_lm, pwi_img, os.path.join(data_dir, "CBF_map_LS_ext.nii.gz"))
		save_nifti(abv_map_lm, pwi_img, os.path.join(data_dir, "ABV_map_LS_ext.nii.gz"))
		save_nifti(att_a_map_lm, pwi_img, os.path.join(data_dir, "ATT_A_map_LS_ext.nii.gz"))

		# Print LS results summary with additional maps
		additional_maps = {'abv': abv_map_lm, 'att_a': att_a_map_lm}
		print_ls_summary(att_map_lm, cbf_map_lm, additional_maps)

	# Set parameter information to save in csv file
	parameter_info = {
		'tau': tau,
		'echo_time': echo_time,
		'lambd': param_config.get('lambd', 0.9),
		'a': param_config.get('a', 0.68),
		'T1a': param_config.get('T1a', 1.65),
		'T1': param_config.get('T1', 1.6),
		'fitting_method': fitting_method
	}

	# Save results and parameter information to csv file
	csv_path_summary = save_ls_results_summary_csv(
		att_map_lm, cbf_map_lm, data_dir
	)

	# 5. === Bayesian fitting ===
	print("\n=== Bayesian Fitting Configuration ===")
	print(f"Maximum number of voxels: {bayesian_config['max_voxels']}")
	print(f"Selection method: {bayesian_config['selection_method']}")
	if bayesian_config['selection_method'] == 'grid':
		print(f"Grid spacing: {bayesian_config['grid_spacing']}")

	# === Run Bayesian fitting based on method ===
	if fitting_method == 'simple':
		print("\n=== Running Bayesian Fitting with LS-based Priors (simple model) ===")
		bayesian_results = bayesian_fit_volume(
			pwi_data, t, m0_data, tau, param_config,
			att_ls_map=att_map_lm,
			cbf_ls_map=cbf_map_lm
		)

		# File suffix for simple model
		suffix = "Bayes"

	elif fitting_method == 'extended':
		print("\n=== Running Bayesian Fitting with LS-based Priors (extended model) ===")
		bayesian_results = bayesian_fit_volume_ext(
			pwi_data, t, m0_data, tau, param_config,
			att_ls_map=att_map_lm,
			cbf_ls_map=cbf_map_lm,
			abv_ls_map=abv_map_lm,
			att_a_ls_map=att_a_map_lm
		)

		# File suffix for extended model
		suffix = "Bayes_Ext"

	# Save Bayesian results
	if bayesian_results is not None:
		print("\n=== Saving Bayesian Results ===")

		# Save parameter maps
		if bayesian_config['save_maps']:
			save_nifti(bayesian_results['att_map'], pwi_img,
					   os.path.join(data_dir, f"ATT_map_{suffix}.nii.gz"))
			save_nifti(bayesian_results['cbf_map'], pwi_img,
					   os.path.join(data_dir, f"CBF_map_{suffix}.nii.gz"))
			save_nifti(bayesian_results['att_std_map'], pwi_img,
					   os.path.join(data_dir, f"ATT_std_map_{suffix}.nii.gz"))
			save_nifti(bayesian_results['cbf_std_map'], pwi_img,
					   os.path.join(data_dir, f"CBF_std_map_{suffix}.nii.gz"))

			# Save fitted parameter maps (if they exist)
			fitted_params = ['T1a', 'T1', 'lambd', 'abv', 'att_a']
			for param in fitted_params:
				fitted_map_key = f'{param}_fitted_map'
				fitted_std_key = f'{param}_fitted_std_map'

				if fitted_map_key in bayesian_results:
					save_nifti(bayesian_results[fitted_map_key], pwi_img,
							   os.path.join(data_dir, f"{param}_fitted_map_{suffix}.nii.gz"))
					save_nifti(bayesian_results[fitted_std_key], pwi_img,
							   os.path.join(data_dir, f"{param}_fitted_std_map_{suffix}.nii.gz"))

		# Save results as csv file (optional)
		if bayesian_config['save_csv']:
			save_results_to_csv(bayesian_results, data_dir, suffix=suffix)

		# Print results comparison
		print_bayesian_summary_flexible(bayesian_results)
		print_comparison_summary(bayesian_results, att_map_lm, cbf_map_lm)

	print("\nDone!")


if __name__ == "__main__":

	asl(fitting_method='simple')
	asl(fitting_method='extended')
