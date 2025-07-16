import os
import numpy as np
from data_handling import load_nifti_file, load_json_metadata
from fitting_single_stan import ls_fit_volume, bayesian_fit_voxel, extract_posterior_summary
from data_handling import save_nifti


def asl():
	"""
		This ASL script consists of the following steps:
		1. read and print data
		2. analyse data for NaN values, they are ignored during fitting
		3. perform LS fitting and save results
		4. perform Bayesian fitting, either for the complete volume or for an adjustable subset of voxels, and save results

		Output:
		- nifti images of the fitted CBF and ATT data
		- print values in csv file for debugging
		"""

	# 1. Read and print data
	script_dir = os.path.dirname(os.path.abspath(__file__))
	data_dir = os.path.join(script_dir, "..", "data", "1TE")

	pwi_nii_path = os.path.join(data_dir, "PWI4D.nii")
	pwi_json_path = os.path.join(data_dir, "PWI4D.json")
	m0_nii_path = os.path.join(data_dir, "M0.nii")
	m0_json_path = os.path.join(data_dir, "M0.json")

	print("Loading data...")
	pwi_img, pwi_data_full = load_nifti_file(pwi_nii_path)
	m0_img, m0_data_full = load_nifti_file(m0_nii_path)
	pwi_meta = load_json_metadata(pwi_json_path)
	m0_meta = load_json_metadata(m0_json_path)

	echo_times = np.array(pwi_meta["EchoTime"])
	plds = np.array(pwi_meta["PostLabelingDelay"])

	# if tau is given as an array, take the first value, else use scalar
	tau_raw = pwi_meta.get("LabelingDuration", 1.8)
	if isinstance(tau_raw, list):
		tau = tau_raw[0]  # take first value
	else:
		tau = tau_raw

	if m0_data_full.ndim == 4 and m0_data_full.shape[3] == 1:
		m0_data = m0_data_full[:, :, :, 0]
	else:
		m0_data = m0_data_full

	chosen_te = 0.01302  # still TBD: will be read from the json file
	indices = [i for i, te in enumerate(echo_times) if np.isclose(te, chosen_te)]

	pwi_data = pwi_data_full[..., indices]
	t = plds[indices]

	print(f"PWI data shape: {pwi_data.shape}")
	print(f"M0 data shape: {m0_data.shape}")
	print(f"Time points: {t}")
	print(f"Tau: {tau}")

	print(f"Original PLDs: {plds}")
	print(f"Chosen indices: {indices}")
	print(f"Filtered time points (t): {t}")
	print(f"Tau (labeling duration): {tau}")

	# 2. Analyse data for NaN
	# Check for NaN in PWI4D.nii
	nan_count = np.sum(np.isnan(pwi_data))
	total_elements = np.prod(pwi_data.shape)
	print(f"NaN values in PWI data: {nan_count}/{total_elements} ({100 * nan_count / total_elements:.2f}%)")

	if nan_count > 0:
		print("PWI data contains NaN values, which is filtered out during the fitting process")

	# Check for NaN in M0 data
	m0_nan_count = np.sum(np.isnan(m0_data))
	m0_total_elements = np.prod(m0_data.shape)
	print(f"NaN values in M0 data: {m0_nan_count}/{m0_total_elements} ({100 * m0_nan_count / m0_total_elements:.2f}%)")

	if m0_nan_count > 0:
		print("M0 data contains NaN values! NaN values will be filtered out during fitting process")

	# Check data range (excluding NaN values)
	pwi_finite = pwi_data[np.isfinite(pwi_data)]
	m0_finite = m0_data[np.isfinite(m0_data)]

	if len(pwi_finite) > 0:
		print(f"PWI data range: [{np.min(pwi_finite):.6f}, {np.max(pwi_finite):.6f}]")
	else:
		print("WARNING: No finite values in PWI data!")

	if len(m0_finite) > 0:
		print(f"M0 data range: [{np.min(m0_finite):.6f}, {np.max(m0_finite):.6f}]")
	else:
		print("WARNING: No finite values in M0 data!")

	print(f"Using {len(indices)} volumes with TE = {chosen_te}")

	# 3. LS fitting
	print("Running LM fitting ...")
	att_map_lm, cbf_map_lm = ls_fit_volume(pwi_data, t, m0_data, tau)

	print("Saving LM results ...")
	save_nifti(att_map_lm, pwi_img, os.path.join(data_dir, "ATT_map_LM.nii.gz"))
	save_nifti(cbf_map_lm, pwi_img, os.path.join(data_dir, "CBF_map_LM.nii.gz"))

	# 4. Bayesian fitting
	print("Running Bayesian fitting for multiple voxels...")

	# Configuration for subset fitting
	# Different options for voxel selection for debugging
	bayesian_config = {
		'max_voxels': 1,  # Maximum number of voxels to fit (None = all valid voxels)
		'selection_method': 'grid',  # 'random', 'grid', 'best_lm', 'all'
		'grid_spacing': 5,  # For grid selection
		'save_maps': True,  # Save parameter maps
	}

	# Run Bayesian fitting on subset
	bayesian_results = bayesian_fit_subset(
		pwi_data, t, m0_data, tau,
		att_map_lm, cbf_map_lm,
		bayesian_config
	)

	if bayesian_results is not None:
		print("Saving Bayesian results...")

		# Save parameter maps if requested
		if bayesian_config['save_maps']:
			save_nifti(bayesian_results['att_map'], pwi_img,
					   os.path.join(data_dir, "ATT_map_Bayesian.nii.gz"))
			save_nifti(bayesian_results['cbf_map'], pwi_img,
					   os.path.join(data_dir, "CBF_map_Bayesian.nii.gz"))
			save_nifti(bayesian_results['att_std_map'], pwi_img,
					   os.path.join(data_dir, "ATT_std_map_Bayesian.nii.gz"))
			save_nifti(bayesian_results['cbf_std_map'], pwi_img,
					   os.path.join(data_dir, "CBF_std_map_Bayesian.nii.gz"))

		# Print summary of results
		print_bayesian_summary(bayesian_results)

	print("Done!")


def bayesian_fit_subset(pwi_data, t, m0_data, tau, att_map_lm, cbf_map_lm, config):
	"""
	Perform Bayesian fitting on a subset of voxels

	Parameters:
	- pwi_data: 4D PWI data
	- t: time points
	- m0_data: M0 data
	- tau: labeling duration
	- att_map_lm: ATT map from LM fitting (used for voxel selection)
	- cbf_map_lm: CBF map from LM fitting (used for voxel selection)

	"""

	# Select voxels based on configuration
	selected_voxels = select_voxels_for_bayesian(
		att_map_lm, cbf_map_lm, pwi_data, m0_data, config
	)

	if len(selected_voxels) == 0:
		print("No suitable voxels found for Bayesian fitting")
		return None

	print(f"Selected {len(selected_voxels)} voxels for Bayesian fitting")

	# Initialize result arrays
	shape = pwi_data.shape[:3]
	att_map = np.full(shape, np.nan)
	cbf_map = np.full(shape, np.nan)
	att_std_map = np.full(shape, np.nan)
	cbf_std_map = np.full(shape, np.nan)

	# Store individual results
	individual_results = []

	# Process each selected voxel
	lambd = 0.9
	successful_fits = 0
	failed_fits = 0

	for i, (x, y, z) in enumerate(selected_voxels):
		print(f"Processing voxel {i + 1}/{len(selected_voxels)}: ({x}, {y}, {z})")

		# Get data for this selected voxel
		signal = pwi_data[x, y, z, :]
		m0 = m0_data[x, y, z]

		# Filter out NaN values
		valid_mask = np.isfinite(signal) & (signal != 0)
		if np.sum(valid_mask) < 2:
			print(f"  Skipping voxel ({x}, {y}, {z}): insufficient valid data points")
			failed_fits += 1
			continue

		signal_clean = signal[valid_mask]
		t_clean = t[valid_mask]

		# Calculate m0a
		m0a = m0 / (6000 * lambd)

		try:
			# Run Bayesian fitting
			fit = bayesian_fit_voxel(t_clean, signal_clean, m0a, tau)

			if fit is not None:
				# Extract results
				summary = extract_posterior_summary(fit)

				if summary is not None:
					# Store in maps
					att_map[x, y, z] = summary['att_mean']
					cbf_map[x, y, z] = summary['cbf_mean']
					att_std_map[x, y, z] = summary['att_std']
					cbf_std_map[x, y, z] = summary['cbf_std']

					# Store individual results
					individual_result = {
						'voxel': (x, y, z),
						'att_mean': summary['att_mean'],
						'att_std': summary['att_std'],
						'cbf_mean': summary['cbf_mean'],
						'cbf_std': summary['cbf_std'],
						'att_lm': att_map_lm[x, y, z],
						'cbf_lm': cbf_map_lm[x, y, z]
					}
					individual_results.append(individual_result)

					successful_fits += 1
					print(f"  Success: ATT={summary['att_mean']:.3f}±{summary['att_std']:.3f}s, "
						  f"CBF={summary['cbf_mean']:.1f}±{summary['cbf_std']:.1f} ml/min/100g")
				else:
					print(f"  Failed to extract summary for voxel ({x}, {y}, {z})")
					failed_fits += 1
			else:
				print(f"  Bayesian fitting failed for voxel ({x}, {y}, {z})")
				failed_fits += 1

		except Exception as e:
			print(f"  Error fitting voxel ({x}, {y}, {z}): {e}")
			failed_fits += 1
			continue

	# Print comprehensive summary
	print(f"\n=== Bayesian Fitting Summary ===")
	print(f"Total valid voxels available: {len(selected_voxels)}")
	print(
		f"Successfully fitted: {successful_fits}/{len(selected_voxels)} ({100 * successful_fits / len(selected_voxels):.1f}%)")
	print(f"Failed fits: {failed_fits}/{len(selected_voxels)} ({100 * failed_fits / len(selected_voxels):.1f}%)")

	return {
		'att_map': att_map,
		'cbf_map': cbf_map,
		'att_std_map': att_std_map,
		'cbf_std_map': cbf_std_map,
		'individual_results': individual_results,
		'selected_voxels': selected_voxels,
		'successful_fits': successful_fits,
		'failed_fits': failed_fits,
		'total_valid_voxels': len(selected_voxels)
	}


def select_voxels_for_bayesian(att_map_lm, cbf_map_lm, pwi_data, m0_data, config):
	"""
	Select voxels for Bayesian fitting based on configuration
	"""

	valid_mask = (
			~np.isnan(att_map_lm) &
			~np.isnan(cbf_map_lm)
	)

	# Count valid voxels
	initial_valid_count = np.sum(valid_mask)
	print(f"Valid voxels (non-NaN): {initial_valid_count}")

	# Additional check for sufficient data quality (always applied)
	shape = pwi_data.shape[:3]
	data_quality_excluded = 0

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
						data_quality_excluded += 1

	# Find all valid coordinates
	valid_coords = np.where(valid_mask)
	all_valid_voxels = list(zip(valid_coords[0], valid_coords[1], valid_coords[2]))

	print(f"Final valid voxels for Bayesian fitting: {len(all_valid_voxels)}")

	if len(all_valid_voxels) == 0:
		return []

	# Select subset based on method
	max_voxels = config['max_voxels']
	selection_method = config['selection_method']

	if selection_method == 'all':
		# Use all valid voxels
		selected_voxels = all_valid_voxels
		print(f"Using all {len(selected_voxels)} valid voxels")

	elif selection_method == 'random':
		if max_voxels is None or len(all_valid_voxels) <= max_voxels:
			selected_voxels = all_valid_voxels
		else:
			# Randomly select voxels
			np.random.seed(42)  # For reproducibility
			selected_indices = np.random.choice(len(all_valid_voxels), max_voxels, replace=False)
			selected_voxels = [all_valid_voxels[i] for i in selected_indices]
			print(f"Randomly selected {len(selected_voxels)} out of {len(all_valid_voxels)} valid voxels")

	elif selection_method == 'grid':
		# Select voxels on a regular grid
		spacing = config.get('grid_spacing', 5)
		selected_voxels = []

		for x, y, z in all_valid_voxels:
			if (x % spacing == 0 and y % spacing == 0 and z % spacing == 0):
				selected_voxels.append((x, y, z))
				if max_voxels is not None and len(selected_voxels) >= max_voxels:
					break

		print(f"Grid selection (spacing={spacing}): {len(selected_voxels)} out of {len(all_valid_voxels)} valid voxels")

	elif selection_method == 'best_lm':
		# Select voxels with best LM fitting result
		# For now, select voxels closest to median ATT and CBF values
		valid_att = att_map_lm[valid_coords]
		valid_cbf = cbf_map_lm[valid_coords]

		median_att = np.median(valid_att)
		median_cbf = np.median(valid_cbf)

		# Calculate distances to median
		distances = np.sqrt((valid_att - median_att) ** 2 + (valid_cbf - median_cbf) ** 2)

		# Select voxels with smallest distances
		sorted_indices = np.argsort(distances)
		if max_voxels is None:
			selected_voxels = [all_valid_voxels[i] for i in sorted_indices]
		else:
			selected_voxels = [all_valid_voxels[i] for i in sorted_indices[:max_voxels]]

		print(f"Best LM selection: {len(selected_voxels)} out of {len(all_valid_voxels)} valid voxels")

	else:
		# Default for using all voxels
		selected_voxels = all_valid_voxels
		print(f"Using all {len(selected_voxels)} valid voxels")

	return selected_voxels


def save_results_to_csv(results, data_dir):
	"""
	Save fitting results to CSV file
	"""
	import csv

	csv_path = os.path.join(data_dir, "results.csv")

	with open(csv_path, 'w', newline='') as csvfile:
		fieldnames = ['x', 'y', 'z', 'att_mean', 'att_std', 'cbf_mean', 'cbf_std', 'att_lm', 'cbf_lm']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		writer.writeheader()
		for result in results['individual_results']:
			x, y, z = result['voxel']
			writer.writerow({
				'x': x,
				'y': y,
				'z': z,
				'att_mean': result['att_mean'],
				'att_std': result['att_std'],
				'cbf_mean': result['cbf_mean'],
				'cbf_std': result['cbf_std'],
				'att_lm': result['att_lm'],
				'cbf_lm': result['cbf_lm']
			})

	print(f"Saved individual results to {csv_path}")


def print_bayesian_summary(results):
	"""
	Print summary of Bayesian fitting results
	"""
	individual_results = results['individual_results']

	if len(individual_results) == 0:
		print("No successful Bayesian fits to summarise")
		return

	# Extract values
	att_means = [r['att_mean'] for r in individual_results]
	att_stds = [r['att_std'] for r in individual_results]
	cbf_means = [r['cbf_mean'] for r in individual_results]
	cbf_stds = [r['cbf_std'] for r in individual_results]

	print(f"\nBayesian Parameter Summary ({len(individual_results)} voxels):")
	print(f"ATT (mean): {np.mean(att_means):.3f} ± {np.std(att_means):.3f}s")
	print(f"ATT (range): [{np.min(att_means):.3f}, {np.max(att_means):.3f}]s")
	print(f"ATT uncertainty (mean): {np.mean(att_stds):.3f}s")

	print(f"CBF (mean): {np.mean(cbf_means):.1f} ± {np.std(cbf_means):.1f} ml/min/100g")
	print(f"CBF (range): [{np.min(cbf_means):.1f}, {np.max(cbf_means):.1f}] ml/min/100g")
	print(f"CBF uncertainty (mean): {np.mean(cbf_stds):.1f} ml/min/100g")



if __name__ == "__main__":
	asl()