import os
import numpy as np
from data_handling import load_nifti_file, load_json_metadata, save_nifti
from fitting_single_te import create_parameter_config, convert_parameter
from fitting_single_te import ls_fit_volume
from fitting_single_te import bayesian_fit_subset, print_bayesian_summary_flexible, print_comparison_summary
from csv_utils import save_results_to_csv, save_ls_results_summary_csv

def asl():
	"""
			This ASL script consists of the following steps:
			1. read and print data
			2. analyse data for NaN values, they are ignored during fitting
			3. perform Least Squares fitting and save results
			4. perform Bayesian fitting, either for the complete volume or for an adjustable subset of voxels, and save results

			Flexible parameter inputs (as nifti images, nparray, scalar) are possible,
			also the possibility to use voxelwise values from LS fitting as priors for Bayesian fitting.
			Further, results of fitting algorithms can be printed into a csv file.

			Output:
			- nifti images of the fitted CBF and ATT data
			- print values in csv file for debugging
			"""

	# 1. === read and print data ===
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

	echo_time = echo_times[0]  # since we use single-te data, we simply choose the first value
	indices = [i for i, te in enumerate(echo_times) if np.isclose(te, echo_time)]
	pwi_data = pwi_data_full[..., indices]
	t = plds[indices]

	print(f"Using TE = {echo_time} s, tau = {tau} s")
	print(f"PWI shape: {pwi_data.shape}, M0 shape: {m0_data.shape}")

	# 2. === analyse data for NaN values ===
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

	# 3. === Least Squares fitting ===
	#  Parameter configuration
	print("\n=== Parameter Configuration ===")
	param_config = create_parameter_config()
	param_config['T1_blood'] = 1.65  # or "path/to/T1_blood_map.nii"
	param_config['T1_tissue'] = 1.3  # or "path/to/T1_tissue_map.nii"
	param_config['lambda_blood'] = 0.9
	param_config['alpha'] = 0.85 * 0.8

	# Configure LS-based priors
	param_config['att_prior_from_ls'] = True  # Use LS ATT results as priors
	param_config['cbf_prior_from_ls'] = True  # Use LS CBF results as priors
	param_config['att_prior_std'] = 0.3  # Standard deviation for ATT prior
	param_config['cbf_prior_std'] = 15.0  # Standard deviation for CBF prior

	print(f"Using LS-based priors: ATT={param_config['att_prior_from_ls']}, CBF={param_config['cbf_prior_from_ls']}")
	print(f"Prior standard deviations: ATT={param_config['att_prior_std']}, CBF={param_config['cbf_prior_std']}")
	# for enabling fitting T1_blood as free parameter: set param_config['fit_T1_blood'] = True

	# Set parameter maps
	shape = pwi_data.shape[:3]
	print("Set parameter maps...")
	T1_blood_map = convert_parameter(param_config['T1_blood'], shape, "T1_blood")
	T1_tissue_map = convert_parameter(param_config['T1_tissue'], shape, "T1_tissue")
	lambda_map = convert_parameter(param_config['lambda_blood'], shape, "lambda_blood")
	alpha_map = convert_parameter(param_config['alpha'], shape, "alpha")
	param_maps = {
		'T1_blood': T1_blood_map,
		'T1_tissue': T1_tissue_map,
		'lambda': lambda_map,
		'alpha': alpha_map
	}

	# === Least Squares fitting ===
	print("\n=== Running Least Squares fitting ===")
	att_map_lm, cbf_map_lm = ls_fit_volume(pwi_data, t, m0_data, tau)

	print("Saving LS results...")
	save_nifti(att_map_lm, pwi_img, os.path.join(data_dir, "ATT_map_LS.nii.gz"))
	save_nifti(cbf_map_lm, pwi_img, os.path.join(data_dir, "CBF_map_LS.nii.gz"))

	def print_ls_summary(att_map_lm, cbf_map_lm):
		"""
		Print summary of Least Squares fitting results
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

	# Print LS results summary
	print_ls_summary(att_map_lm, cbf_map_lm)

	# Set parameter information to save in csv file
	parameter_info = {
		'tau': tau,
		'echo_time': echo_time,
		'lambda_blood': param_config.get('lambda_blood', 0.9),
		'alpha': param_config.get('alpha', 0.85 * 0.8),
		'T1_blood': param_config.get('T1_blood', 1.65),
		'T1_tissue': param_config.get('T1_tissue', 1.3)
	}

	# Save results and parameter information to csv file
	csv_path_summary = save_ls_results_summary_csv(
		att_map_lm, cbf_map_lm, data_dir
	)

	# 4. === Bayesian fitting ===
	print("\n=== Bayesian Fitting Configuration ===")
	bayesian_config = {
		'max_voxels': 1000,  # Maximum number of voxels to fit (None = fitting all voxels)
		'selection_method': 'grid',  # 'random', 'grid', 'best_lm', 'all'
		'grid_spacing': 5,  # For grid selection
		'save_maps': True,  # Save parameter maps
		'save_csv': True,  # Save results to a CSV file
	}

	print(f"Maximum number of voxels: {bayesian_config['max_voxels']}")
	print(f"Selection method: {bayesian_config['selection_method']}")

	# === Run Bayesian fitting with LS priors ===
	print("\n=== Running Bayesian Fitting with LS-based Priors ===")
	bayesian_results = bayesian_fit_subset(
		pwi_data, t, m0_data, tau,
		param_maps, param_config,
		att_map_lm, cbf_map_lm,
		bayesian_config
	)

	if bayesian_results is not None:
		print("\n=== Saving Bayesian Results ===")

		# Save parameter maps
		if bayesian_config['save_maps']:
			save_nifti(bayesian_results['att_map'], pwi_img,
					   os.path.join(data_dir, "ATT_map_Bayes_Flex.nii.gz"))
			save_nifti(bayesian_results['cbf_map'], pwi_img,
					   os.path.join(data_dir, "CBF_map_Bayes_Flex.nii.gz"))
			save_nifti(bayesian_results['att_std_map'], pwi_img,
					   os.path.join(data_dir, "ATT_std_map_Bayes_Flex.nii.gz"))
			save_nifti(bayesian_results['cbf_std_map'], pwi_img,
					   os.path.join(data_dir, "CBF_std_map_Bayes_Flex.nii.gz"))

			# Save fitted parameter maps
			if 'T1_blood_fitted_map' in bayesian_results:
				save_nifti(bayesian_results['T1_blood_fitted_map'], pwi_img,
						   os.path.join(data_dir, "T1_blood_fitted_map.nii.gz"))
				save_nifti(bayesian_results['T1_blood_fitted_std_map'], pwi_img,
						   os.path.join(data_dir, "T1_blood_fitted_std_map.nii.gz"))

			if 'T1_tissue_fitted_map' in bayesian_results:
				save_nifti(bayesian_results['T1_tissue_fitted_map'], pwi_img,
						   os.path.join(data_dir, "T1_tissue_fitted_map.nii.gz"))
				save_nifti(bayesian_results['T1_tissue_fitted_std_map'], pwi_img,
						   os.path.join(data_dir, "T1_tissue_fitted_std_map.nii.gz"))

			if 'lambda_fitted_map' in bayesian_results:
				save_nifti(bayesian_results['lambda_fitted_map'], pwi_img,
						   os.path.join(data_dir, "lambda_fitted_map.nii.gz"))
				save_nifti(bayesian_results['lambda_fitted_std_map'], pwi_img,
						   os.path.join(data_dir, "lambda_fitted_std_map.nii.gz"))

		# Save results as csv file
		if bayesian_config['save_csv']:
			save_results_to_csv(bayesian_results, data_dir)

		# Print results comparison
		print_bayesian_summary_flexible(bayesian_results)
		print_comparison_summary(bayesian_results, att_map_lm, cbf_map_lm)

	print("\nDone!")


if __name__ == "__main__":
	asl()