# Debugging function fit for one voxel in least squares fitting
"""
# Debug fit for one voxel
Use in asl_single_te.py after the configuration and before the LS fitting to debug one voxel only
	from voxel import debug_ls_fit_m0_variations, debug_bayesian_fit_voxel
	#results = debug_ls_fit_m0_variations(1,1,1, pwi_data, m0_data, t, tau, param_config)
	results = debug_bayesian_fit_voxel(57, 60, 4, pwi_data, m0_data, t, tau, param_config)
"""

# Debugging function for subset of voxels in bayesian fitting
import numpy as np
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

	# Initialize result arrays
	shape = pwi_data.shape[:3]
	att_map = np.full(shape, np.nan)
	cbf_map = np.full(shape, np.nan)
	att_std_map = np.full(shape, np.nan)
	cbf_std_map = np.full(shape, np.nan)

	# Initialize fitted parameter maps if needed
	fitted_param_maps = {}
	if param_config['fit_T1a']:
		fitted_param_maps['T1a_fitted_map'] = np.full(shape, np.nan)
		fitted_param_maps['T1a_fitted_std_map'] = np.full(shape, np.nan)
	if param_config['fit_T1']:
		fitted_param_maps['T1_fitted_map'] = np.full(shape, np.nan)
		fitted_param_maps['T1_fitted_std_map'] = np.full(shape, np.nan)
	if param_config['fit_lambd']:
		fitted_param_maps['lambd_fitted_map'] = np.full(shape, np.nan)
		fitted_param_maps['lambd_fitted_std_map'] = np.full(shape, np.nan)

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
		T1a_val = param_maps['T1a'][x, y, z]
		T1_val = param_maps['T1'][x, y, z]
		lambd_val = param_maps['lambd'][x, y, z]
		a_val = param_maps['a'][x, y, z]

		# Skip if any parameter is NaN
		if (np.isnan(T1a_val) or np.isnan(T1_val) or
				np.isnan(lambd_val) or np.isnan(a_val)):
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

		# Calculate M0a
		M0a = 1.0

		try:
			# Run Bayesian fitting with flexible parameters and LS priors
			from fitting_single_te import bayesian_fit_voxel
			fit = bayesian_fit_voxel(
				t_clean, signal_clean, M0a, tau,
				T1a_val, T1_val, lambd_val, a_val,
				param_config, att_ls_val, cbf_ls_val
			)

			if fit is not None:
				# Extract results
				att_samples = fit['att'].flatten()
				cbf_samples = fit['cbf'].flatten()  # Already in ml/min/100g

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
				if param_config['fit_T1a']:
					T1a_samples = fit['T1a_param'].flatten()
					T1a_fitted_mean = np.mean(T1a_samples)
					T1a_fitted_std = np.std(T1a_samples)
					fitted_param_maps['T1a_fitted_map'][x, y, z] = T1a_fitted_mean
					fitted_param_maps['T1a_fitted_std_map'][x, y, z] = T1a_fitted_std
					fitted_results['T1a_fitted_mean'] = T1a_fitted_mean
					fitted_results['T1a_fitted_std'] = T1a_fitted_std

				if param_config['fit_T1']:
					T1_samples = fit['T1_param'].flatten()
					T1_fitted_mean = np.mean(T1_samples)
					T1_fitted_std = np.std(T1_samples)
					fitted_param_maps['T1_fitted_map'][x, y, z] = T1_fitted_mean
					fitted_param_maps['T1_fitted_std_map'][x, y, z] = T1_fitted_std
					fitted_results['T1_fitted_mean'] = T1_fitted_mean
					fitted_results['T1_fitted_std'] = T1_fitted_std

				if param_config['fit_lambd']:
					lambd_samples = fit['lambd_param'].flatten()
					lambd_fitted_mean = np.mean(lambd_samples)
					lambd_fitted_std = np.std(lambd_samples)
					fitted_param_maps['lambd_fitted_map'][x, y, z] = lambd_fitted_mean
					fitted_param_maps['lambd_fitted_std_map'][x, y, z] = lambd_fitted_std
					fitted_results['lambd_fitted_mean'] = lambd_fitted_mean
					fitted_results['lambd_fitted_std'] = lambd_fitted_std

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
						'T1a': T1a_val,
						'T1': T1_val,
						'lambd': lambd_val,
						'a': a_val
					},
					'fitted_parameters': fitted_results
				}
				individual_results.append(individual_result)

				successful_fits += 1
				print(f"  Success: ATT={att_mean:.3f}±{att_std:.3f}s, "
					  f"CBF={cbf_mean:.1f}±{cbf_std:.1f} ml/min/100g")

				# Show comparison with LS
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
