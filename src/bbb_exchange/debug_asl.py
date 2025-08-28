# Debugging function fit for one voxel in least squares fitting
"""
# Debug fit for one voxel
Use in asl_single_te.py after the configuration and before the LS fitting to debug one voxel only
	from voxel import debug_ls_fit_m0_variations, debug_bayesian_fit_voxel
	#results = debug_ls_fit_m0_variations(1,1,1, pwi_data, m0_data, t, tau, param_config)
	results = debug_bayesian_fit_voxel(57, 60, 4, pwi_data, m0_data, t, tau, param_config)
"""

# Debugging function for subset of voxels in bayesian fitting in single-echo time
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


# debug functions for multi-echo time
# insert into asl_multi_te.py
"""
import matplotlib.pyplot as plt
import os
def run_custom_voxel_bayesian(voxel_coords):

    print(f"\n=== Custom Voxel Bayesian Analysis: {voxel_coords} ===")

    # Load configuration
    multite_config = create_multite_config()

    # Load data (simplified version for single voxel)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data", "multite")

    pwi_path = os.path.join(data_dir, "PWI4D.nii")
    pwi_json_path = os.path.join(data_dir, "PWI4D.json")
    m0_path = os.path.join(data_dir, "M0.nii.gz")

    pwi_img, pwi_data_full = load_nifti_file(pwi_path)
    m0_img, m0_data_full = load_nifti_file(m0_path)
    pwi_meta = load_json_metadata(pwi_json_path)

    # Handle M0 data
    if m0_data_full.ndim == 4 and m0_data_full.shape[3] == 1:
        m0_data = m0_data_full[:, :, :, 0]
    else:
        m0_data = m0_data_full

    # Prepare Multi-TE data
    tis, tes, ntes, taus = prepare_multite_data(pwi_meta)

    # Extract data for this specific voxel
    x, y, z = voxel_coords
    signal = pwi_data_full[x, y, z, :]
    m0 = m0_data[x, y, z]

    # Check for valid data
    if (np.any(np.isnan(signal)) or np.any(np.isinf(signal)) or
            np.all(signal == 0) or np.isnan(m0) or np.isinf(m0) or m0 <= 0):
        print(f"Invalid data at voxel ({x}, {y}, {z})")
        return None, None, None, None

    # Normalize signal using config values
    signal_normalized = signal / (m0 * config['normalization']['m0_multiplier'])
    M0a = m0 / (config['normalization']['m0a_scale'] * 0.9)

    print(f"Signal range: {np.min(signal_normalized):.6e} to {np.max(signal_normalized):.6e}")
    print(f"M0a: {M0a:.6e}")

    # Optional: Get LS fit results for priors (you can comment this out if you want pure Bayesian)
    print("Getting LS fit for priors...")
    att_ls, cbf_ls, rmse_ls = ls_fit_voxel_multite(
        tis, tes, ntes, signal_normalized, M0a, taus,
        T1=multite_config['T1'],
        T1a=multite_config['T1a'],
        T2=multite_config['T2'],
        T2a=multite_config['T2a'],
        texch=multite_config['texch'],
        itt=multite_config['itt'],
        lambd=multite_config['lambd'],
        alpha=multite_config['alpha']
    )

    if not np.isnan(att_ls) and not np.isnan(cbf_ls):
        print(f"LS priors: ATT = {att_ls:.3f}s, CBF = {cbf_ls:.1f} ml/min/100g, RMSE = {rmse_ls:.6f}")
    else:
        print("LS fitting failed, using default priors")
        att_ls, cbf_ls = None, None

    # Create Bayesian configuration
    bayesian_config = create_multite_bayesian_config()
    bayesian_config.update(multite_config)  # Use same physiological parameters

    # Run Bayesian fitting for this single voxel
    print("Running Bayesian fitting...")
    fit = bayesian_fit_voxel_multite(
        tis, tes, ntes, signal_normalized, M0a, taus,
        bayesian_config,
        att_ls_val=att_ls,
        cbf_ls_val=cbf_ls
    )

    bayes_results = None
    if fit is not None:
        try:
            df = fit.to_frame()
            att_mean = df['att'].mean()
            att_std = df['att'].std()
            cbf_mean = df['cbf'].mean()
            cbf_std = df['cbf'].std()

            print(f"Bayesian results - ATT: {att_mean:.3f}±{att_std:.3f}s, "
                  f"CBF: {cbf_mean:.1f}±{cbf_std:.1f} ml/min/100g")

            bayes_results = (att_mean, cbf_mean, att_std, cbf_std)

            # Print some sample statistics
            print(f"ATT samples: min={df['att'].min():.3f}, max={df['att'].max():.3f}")
            print(f"CBF samples: min={df['cbf'].min():.1f}, max={df['cbf'].max():.1f}")

        except Exception as e:
            print(f"Error extracting Bayesian results: {e}")
            fit = None
    else:
        print("Bayesian fitting failed")

    # Generate fitted signals for plotting
    fitted_signal_ls = None
    if not np.isnan(att_ls) and not np.isnan(cbf_ls):
        fitted_signal_ls = deltaM_multite_model(
            tis, tes, ntes, att_ls, cbf_ls, M0a, taus,
            t1=multite_config['T1'], t1b=multite_config['T1a'],
            t2=multite_config['T2'], t2b=multite_config['T2a'],
            texch=multite_config['texch'], itt=multite_config['itt'],
            lambd=multite_config['lambd'], alpha=multite_config['alpha']
        )

    fitted_signal_bayes = None
    if bayes_results is not None:
        fitted_signal_bayes = deltaM_multite_model(
            tis, tes, ntes, bayes_results[0], bayes_results[1], M0a, taus,
            t1=multite_config['T1'], t1b=multite_config['T1a'],
            t2=multite_config['T2'], t2b=multite_config['T2a'],
            texch=multite_config['texch'], itt=multite_config['itt'],
            lambd=multite_config['lambd'], alpha=multite_config['alpha']
        )

    # Create enhanced plot with Bayesian results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: All measurements
    measurement_indices = np.arange(len(signal_normalized))
    ax1.plot(measurement_indices, signal_normalized, 'bo-',
             linewidth=2, markersize=6, label='Measured data')

    if fitted_signal_ls is not None:
        ax1.plot(measurement_indices, fitted_signal_ls, 'r-',
                 linewidth=2, label='LS fit (prior)')

    if fitted_signal_bayes is not None:
        ax1.plot(measurement_indices, fitted_signal_bayes, 'g--',
                 linewidth=2, label='Bayesian fit')

    ax1.set_xlabel('Measurement Index')
    ax1.set_ylabel('Normalized Signal')

    title = f'Multi-TE ASL Bayesian Fit - Voxel ({x}, {y}, {z})\n'
    if not np.isnan(att_ls) and not np.isnan(cbf_ls):
        title += f'LS: ATT={att_ls:.3f}s, CBF={cbf_ls:.1f} ml/min/100g\n'
    if bayes_results is not None:
        title += f'Bayes: ATT={bayes_results[0]:.3f}±{bayes_results[2]:.3f}s, '
        title += f'CBF={bayes_results[1]:.1f}±{bayes_results[3]:.1f} ml/min/100g'

    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Signal vs TI for different TEs
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(tes))))
    te_unique = np.unique(tes)

    for i, te_val in enumerate(te_unique):
        te_indices = []
        current_idx = 0
        for j, ti_val in enumerate(tis):
            for k in range(ntes[j]):
                if np.isclose(tes[current_idx], te_val):
                    te_indices.append(current_idx)
                current_idx += 1

        if te_indices:
            te_tis, te_signals, te_fitted_ls, te_fitted_bayes = [], [], [], []
            current_idx = 0
            for j, ti_val in enumerate(tis):
                for k in range(ntes[j]):
                    if current_idx in te_indices:
                        te_tis.append(ti_val)
                        te_signals.append(signal_normalized[current_idx])
                        if fitted_signal_ls is not None:
                            te_fitted_ls.append(fitted_signal_ls[current_idx])
                        if fitted_signal_bayes is not None:
                            te_fitted_bayes.append(fitted_signal_bayes[current_idx])
                    current_idx += 1

            ax2.plot(te_tis, te_signals, 'o', color=colors[i],
                     markersize=6, label=f'TE = {te_val * 1000:.1f} ms (data)')

            if fitted_signal_ls is not None and te_fitted_ls:
                ax2.plot(te_tis, te_fitted_ls, '-', color=colors[i],
                         linewidth=1, alpha=0.6)

            if fitted_signal_bayes is not None and te_fitted_bayes:
                ax2.plot(te_tis, te_fitted_bayes, '--', color=colors[i],
                         linewidth=2, alpha=0.8)

    ax2.set_xlabel('Inversion Time (s)')
    ax2.set_ylabel('Normalized Signal')
    ax2.set_title('Signal vs TI for different TEs')

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='black', linestyle='',
                              markersize=6, label='Measured data')]
    if fitted_signal_ls is not None:
        legend_elements.append(Line2D([0], [0], color='black', linestyle='-',
                                      linewidth=1, alpha=0.6, label='LS fit'))
    if fitted_signal_bayes is not None:
        legend_elements.append(Line2D([0], [0], color='black', linestyle='--',
                                      linewidth=2, label='Bayesian fit'))

    ax2.legend(handles=legend_elements)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(data_dir, f"voxel_{x}_{y}_{z}_bayesian_fit_MultiTE.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.show()

    # Print results summary
    print(f"\n=== Results Summary for Voxel ({x}, {y}, {z}) ===")
    if not np.isnan(att_ls) and not np.isnan(cbf_ls):
        print(f"LS (prior): ATT = {att_ls:.3f}s, CBF = {cbf_ls:.1f} ml/min/100g, RMSE = {rmse_ls:.6f}")
    if bayes_results is not None:
        print(f"Bayesian: ATT = {bayes_results[0]:.3f}±{bayes_results[2]:.3f}s, "
              f"CBF = {bayes_results[1]:.1f}±{bayes_results[3]:.1f} ml/min/100g")

    return att_ls, cbf_ls, rmse_ls, bayes_results



def bayesian_surface_plot(voxel_coords):
    # Bayesian Surface Plot for one voxel

    print(f"\n=== Bayesian Surface Analysis for Voxel {voxel_coords} ===")

    att_mean, cbf_mean, att_std, cbf_std = run_custom_voxel_bayesian(voxel_coords)

    if att_mean is None:
        print("Bayesian fitting failed.")
        return None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data", "multite")
    pwi_path = os.path.join(data_dir, "PWI4D.nii")
    pwi_json_path = os.path.join(data_dir, "PWI4D.json")
    m0_path = os.path.join(data_dir, "M0.nii.gz")

    _, pwi_data = load_nifti_file(pwi_path)
    _, m0_data = load_nifti_file(m0_path)
    pwi_meta = load_json_metadata(pwi_json_path)

    tis, tes, ntes, taus = prepare_multite_data(pwi_meta)
    x, y, z = voxel_coords
    signal = pwi_data[x, y, z, :] / (m0_data[x, y, z] * config['normalization']['m0_multiplier'])
    M0a = m0_data[x, y, z] / (config['normalization']['m0a_scale'] * 0.9)

    fitted_bayes = deltaM_multite_model(
        tis, tes, ntes, att_mean, cbf_mean, M0a, taus
    )

    # 4. --- Surface-Plot ---
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")


    ax.scatter(tis.repeat(ntes), tes, signal, c="blue", label="Measured data")


    ax.plot_trisurf(tis.repeat(ntes), tes, fitted_bayes,
                    color="red", alpha=0.5, label="Bayes fit")

    ax.set_xlabel("Inversion Time TI (s)")
    ax.set_ylabel("Echo Time TE (s)")
    ax.set_zlabel("ΔM (normalized)")
    ax.set_title(f"Bayesian Surface Fit - Voxel {voxel_coords}\n"
                 f"ATT = {att_mean:.3f}±{att_std:.3f}s, "
                 f"CBF = {cbf_mean:.1f}±{cbf_std:.1f} ml/min/100g")

    plt.legend()
    plt.tight_layout()
    plt.show()

    return att_mean, cbf_mean, att_std, cbf_std



def run_surface_analysis_for_voxel(voxel_coords):

    print(f"\n=== Surface Analysis for Voxel {voxel_coords} ===")

    # Load configuration
    multite_config = create_multite_config()

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data", "multite")

    pwi_path = os.path.join(data_dir, "PWI4D.nii")
    pwi_json_path = os.path.join(data_dir, "PWI4D.json")
    m0_path = os.path.join(data_dir, "M0.nii.gz")

    pwi_img, pwi_data_full = load_nifti_file(pwi_path)
    m0_img, m0_data_full = load_nifti_file(m0_path)
    pwi_meta = load_json_metadata(pwi_json_path)

    # Handle M0 data
    if m0_data_full.ndim == 4 and m0_data_full.shape[3] == 1:
        m0_data = m0_data_full[:, :, :, 0]
    else:
        m0_data = m0_data_full

    # Prepare Multi-TE data
    tis, tes, ntes, taus = prepare_multite_data(pwi_meta)

    # Perform LS fitting for the voxel
    x, y, z = voxel_coords
    signal = pwi_data_full[x, y, z, :]
    m0 = m0_data[x, y, z]

    # Check for valid data
    if (np.any(np.isnan(signal)) or np.any(np.isinf(signal)) or
            np.all(signal == 0) or np.isnan(m0) or np.isinf(m0) or m0 <= 0):
        print(f"Invalid data at voxel ({x}, {y}, {z})")
        return None

    # Normalize signal using config values
    signal_normalized = signal / (m0 * config['normalization']['m0_multiplier'])
    M0a = m0 / (config['normalization']['m0a_scale'] * 0.9)

    # Perform LS fitting
    att_fit, cbf_fit, rmse = ls_fit_voxel_multite(
        tis, tes, ntes, signal_normalized, M0a, taus,
        T1=multite_config['T1'],
        T1a=multite_config['T1a'],
        T2=multite_config['T2'],
        T2a=multite_config['T2a'],
        texch=multite_config['texch'],
        itt=multite_config['itt'],
        lambd=multite_config['lambd'],
        alpha=multite_config['alpha']
    )

    if np.isnan(att_fit) or np.isnan(cbf_fit):
        print(f"Fitting failed for voxel ({x}, {y}, {z})")
        return None

    print(f"Fit results: ATT = {att_fit:.3f}s, CBF = {cbf_fit:.1f} ml/min/100g, RMSE = {rmse:.6f}")

    # Create surface visualization
    surface_plot_path = os.path.join(data_dir, f"voxel_{x}_{y}_{z}_surface_analysis.png")
    slices_plot_path = os.path.join(data_dir, f"voxel_{x}_{y}_{z}_surface_slices.png")

    pld_grid, te_grid, signal_grid, fitted_grid, residuals_grid = plot_multite_surface(
        pwi_data_full, tis, tes, ntes, m0_data, taus, voxel_coords,
        att_fit, cbf_fit, output_path=surface_plot_path,
        fixed_view=True,  # Fixe Blickwinkel
        elevation=30, azimuth=45,  # Anpassbare Blickwinkel
        T1=multite_config['T1'], T1a=multite_config['T1a'],
        T2=multite_config['T2'], T2a=multite_config['T2a'],
        texch=multite_config['texch'], itt=multite_config['itt'],
        lambd=multite_config['lambd'], alpha=multite_config['alpha']
    )


	#run_custom_voxel_bayesian([68, 49, 16])
    #results = run_surface_analysis_for_voxel([68, 49, 16])

    #if results:
    #        print(f"\nSurface analysis completed successfully!")
    #        print(f"R-squared: {results['fit_stats']['r_squared']:.4f}")
    #        print(f"RMSE: {results['fit_stats']['rmse']:.6e}")
    #results = run_bayesian_surface_analysis_for_voxel([68, 49, 16])
    
    

def print_multite_bayesian_summary(results):
    #Print summary of Multi-TE Bayesian fitting results
    
    att_map = results['att_map']
    cbf_map = results['cbf_map']
    att_std_map = results['att_std_map']
    cbf_std_map = results['cbf_std_map']

    att_finite = att_map[np.isfinite(att_map)]
    cbf_finite = cbf_map[np.isfinite(cbf_map)]
    att_std_finite = att_std_map[np.isfinite(att_std_map)]
    cbf_std_finite = cbf_std_map[np.isfinite(cbf_std_map)]

    if len(att_finite) > 0:
        print(f"\n=== Multi-TE Bayesian Results Summary ===")
        print(f"ATT (Bayesian): {np.mean(att_finite):.3f} ± {np.std(att_finite):.3f}s")
        print(f"ATT range: [{np.min(att_finite):.3f}, {np.max(att_finite):.3f}]s")
        print(f"ATT uncertainty (mean): {np.mean(att_std_finite):.3f}s")
        print(f"Valid ATT voxels: {len(att_finite)}")

    if len(cbf_finite) > 0:
        print(f"CBF (Bayesian): {np.mean(cbf_finite):.1f} ± {np.std(cbf_finite):.1f} ml/min/100g")
        print(f"CBF range: [{np.min(cbf_finite):.1f}, {np.max(cbf_finite):.1f}] ml/min/100g")
        print(f"CBF uncertainty (mean): {np.mean(cbf_std_finite):.1f} ml/min/100g")
        print(f"Valid CBF voxels: {len(cbf_finite)}")

    print(f"Successful fits: {results['successful_fits']}/{results['total_processed']} "
          f"({100 * results['successful_fits'] / results['total_processed']:.1f}%)")

    # Print fitted parameter summaries
    param_names = ['T1', 'T1a', 'T2', 'T2a', 'texch', 'itt', 'lambd']
    for param_name in param_names:
        fitted_key = f'{param_name}_fitted_map'
        if fitted_key in results:
            fitted_map = results[fitted_key]
            fitted_finite = fitted_map[np.isfinite(fitted_map)]
            if len(fitted_finite) > 0:
                print(f"{param_name} (fitted): {np.mean(fitted_finite):.4f} ± {np.std(fitted_finite):.4f}")


def plot_voxel_fit_multite(pwi_data, tis, tes, ntes, m0_data, taus,
                           voxel_coords, output_path=None,
                           T1=1.3, T1a=1.65, T2=0.050, T2a=0.150,
                           texch=0.1, itt=0.2, lambd=0.9, alpha=0.68):

    from model_multi_te import deltaM_multite_model

    x, y, z = voxel_coords

    # Get signal and M0 for this voxel
    signal = pwi_data[x, y, z, :]
    m0 = m0_data[x, y, z]

    # Check for valid data
    if (np.any(np.isnan(signal)) or np.any(np.isinf(signal)) or
            np.all(signal == 0) or np.isnan(m0) or np.isinf(m0) or m0 <= 0):
        print(f"Invalid data at voxel ({x}, {y}, {z})")
        return None, None, None, None

    # Normalize signal
    signal_normalized = signal / (m0 * 5)
    M0a = m0 / (6000 * 0.9)

    # Perform LS fitting
    print(f"LS fitting voxel ({x}, {y}, {z})...")
    att_fit, cbf_fit, rmse = ls_fit_voxel_multite(
        tis, tes, ntes, signal_normalized, M0a, taus,
        T1=T1, T1a=T1a, T2=T2, T2a=T2a,
        texch=texch, itt=itt, lambd=lambd, alpha=alpha
    )

    if np.isnan(att_fit) or np.isnan(cbf_fit):
        print(f"LS fitting failed for voxel ({x}, {y}, {z})")
        return None, None, None, None

    # Perform Bayesian fitting
    print("Bayesian fitting...")
    bayes_config = create_multite_bayesian_config()
    bayes_config.update({
        "T1": T1, "T1a": T1a, "T2": T2, "T2a": T2a,
        "texch": texch, "itt": itt, "lambd": lambd, "alpha": alpha
    })

    fit = bayesian_fit_voxel_multite(
        tis, tes, ntes, signal_normalized, M0a, taus,
        bayes_config,
        att_ls_val=att_fit,
        cbf_ls_val=cbf_fit
    )

    bayes_results = None
    if fit is not None:
        try:
            df = fit.to_frame()
            att_mean = df['att'].mean()
            att_std = df['att'].std()
            cbf_mean = df['cbf'].mean()
            cbf_std = df['cbf'].std()
            print(f"Bayesian results - ATT: {att_mean:.3f}±{att_std:.3f}s, "
                  f"CBF: {cbf_mean:.1f}±{cbf_std:.1f} ml/min/100g")
            bayes_results = (att_mean, cbf_mean, att_std, cbf_std)
        except Exception as e:
            print(f"Error extracting Bayesian results: {e}")
    else:
        print("Bayesian fitting failed")

    # Generate fitted signals
    fitted_signal_ls = deltaM_multite_model(
        tis, tes, ntes, att_fit, cbf_fit, M0a, taus,
        t1=T1, t1b=T1a, t2=T2, t2b=T2a,
        texch=texch, itt=itt, lambd=lambd, alpha=alpha
    )

    fitted_signal_bayes = None
    if bayes_results is not None:
        fitted_signal_bayes = deltaM_multite_model(
            tis, tes, ntes, bayes_results[0], bayes_results[1], M0a, taus,
            t1=T1, t1b=T1a, t2=T2, t2b=T2a,
            texch=texch, itt=itt, lambd=lambd, alpha=alpha
        )

    # Create enhanced plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: All measurements
    measurement_indices = np.arange(len(signal_normalized))
    ax1.plot(measurement_indices, signal_normalized, 'bo-',
             linewidth=2, markersize=6, label='Measured data')
    ax1.plot(measurement_indices, fitted_signal_ls, 'r-',
             linewidth=2, label='LS fit')

    if fitted_signal_bayes is not None:
        ax1.plot(measurement_indices, fitted_signal_bayes, 'g--',
                 linewidth=2, label='Bayesian fit')

    ax1.set_xlabel('Measurement Index')
    ax1.set_ylabel('Normalized Signal')

    title = f'Multi-TE ASL Fit - Voxel ({x}, {y}, {z})\n'
    title += f'LS: ATT={att_fit:.3f}s, CBF={cbf_fit:.1f} ml/min/100g, RMSE={rmse:.6f}'
    if bayes_results is not None:
        title += f'\nBayes: ATT={bayes_results[0]:.3f}±{bayes_results[2]:.3f}s, '
        title += f'CBF={bayes_results[1]:.1f}±{bayes_results[3]:.1f} ml/min/100g'

    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Signal vs TI for different TEs
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(tes))))
    te_unique = np.unique(tes)

    for i, te_val in enumerate(te_unique):
        te_indices = []
        current_idx = 0
        for j, ti_val in enumerate(tis):
            for k in range(ntes[j]):
                if np.isclose(tes[current_idx], te_val):
                    te_indices.append(current_idx)
                current_idx += 1

        if te_indices:
            te_tis, te_signals, te_fitted_ls, te_fitted_bayes = [], [], [], []
            current_idx = 0
            for j, ti_val in enumerate(tis):
                for k in range(ntes[j]):
                    if current_idx in te_indices:
                        te_tis.append(ti_val)
                        te_signals.append(signal_normalized[current_idx])
                        te_fitted_ls.append(fitted_signal_ls[current_idx])
                        if fitted_signal_bayes is not None:
                            te_fitted_bayes.append(fitted_signal_bayes[current_idx])
                    current_idx += 1

            ax2.plot(te_tis, te_signals, 'o', color=colors[i],
                     markersize=6, label=f'TE = {te_val * 1000:.1f} ms (data)')
            ax2.plot(te_tis, te_fitted_ls, '-', color=colors[i],
                     linewidth=2, alpha=0.8)

            if fitted_signal_bayes is not None:
                ax2.plot(te_tis, te_fitted_bayes, '--', color=colors[i],
                         linewidth=2, alpha=0.6)

    ax2.set_xlabel('Inversion Time (s)')
    ax2.set_ylabel('Normalized Signal')
    ax2.set_title('Signal vs TI for different TEs')

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='black', linestyle='',
                              markersize=6, label='Measured data'),
                       Line2D([0], [0], color='black', linestyle='-',
                              linewidth=2, label='LS fit')]
    if fitted_signal_bayes is not None:
        legend_elements.append(Line2D([0], [0], color='black', linestyle='--',
                                      linewidth=2, label='Bayesian fit'))

    ax2.legend(handles=legend_elements)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    plt.show()

    # Print results
    print(f"\n=== Fit Results for Voxel ({x}, {y}, {z}) ===")
    print(f"LS: ATT = {att_fit:.3f}s, CBF = {cbf_fit:.1f} ml/min/100g, RMSE = {rmse:.6f}")
    if bayes_results is not None:
        print(f"Bayesian: ATT = {bayes_results[0]:.3f}±{bayes_results[2]:.3f}s, "
              f"CBF = {bayes_results[1]:.1f}±{bayes_results[3]:.1f} ml/min/100g")

    return att_fit, cbf_fit, rmse, bayes_results    
    
    
    

def print_multite_ls_summary(att_map, cbf_map, rmse_map):


    att_finite = att_map[np.isfinite(att_map)]
    cbf_finite = cbf_map[np.isfinite(cbf_map)]
    rmse_finite = rmse_map[np.isfinite(rmse_map)]

    if len(att_finite) > 0:
        print(f"\n=== Multi-TE Least Squares Results Summary ===")
        print(f"ATT (LS): {np.mean(att_finite):.3f} ± {np.std(att_finite):.3f}s")
        print(f"ATT range: [{np.min(att_finite):.3f}, {np.max(att_finite):.3f}]s")
        print(f"Valid ATT voxels: {len(att_finite)}")

    if len(cbf_finite) > 0:
        print(f"CBF (LS): {np.mean(cbf_finite):.1f} ± {np.std(cbf_finite):.1f} ml/min/100g")
        print(f"CBF range: [{np.min(cbf_finite):.1f}, {np.max(cbf_finite):.1f}] ml/min/100g")
        print(f"Valid CBF voxels: {len(cbf_finite)}")

    if len(rmse_finite) > 0:
        print(f"RMSE: {np.mean(rmse_finite):.6f} ± {np.std(rmse_finite):.6f}")
        print(f"RMSE range: [{np.min(rmse_finite):.6f}, {np.max(rmse_finite):.6f}]")

def create_surface_data(tis, tes, ntes, signal):

    # Create coordinate arrays for each measurement
    pld_points = []
    te_points = []

    current_idx = 0
    for i, ti_val in enumerate(tis):
        for j in range(ntes[i]):
            pld_points.append(ti_val)
            te_points.append(tes[current_idx])
            current_idx += 1

    pld_points = np.array(pld_points)
    te_points = np.array(te_points)

    # Create regular grid for interpolation
    pld_unique = np.unique(pld_points)
    te_unique = np.unique(te_points)

    pld_grid, te_grid = np.meshgrid(pld_unique, te_unique, indexing='ij')

    # Interpolate signal onto regular grid
    signal_grid = griddata(
        (pld_points, te_points), signal,
        (pld_grid, te_grid),
        method='linear',
        fill_value=np.nan
    )

    return pld_grid, te_grid, signal_grid, pld_points, te_points


def create_fitted_surface(tis, tes, ntes, att, cbf, M0a, taus,
                          T1=1.3, T1a=1.65, T2=0.050, T2a=0.150,
                          texch=0.1, itt=0.2, lambd=0.9, alpha=0.68):

    #Create fitted surface using the multi-TE model


    # Generate fitted signal
    fitted_signal = deltaM_multite_model(
        tis, tes, ntes, att, cbf, M0a, taus,
        t1=T1, t1b=T1a, t2=T2, t2b=T2a,
        texch=texch, itt=itt, lambd=lambd, alpha=alpha
    )

    # Create surface data
    pld_grid, te_grid, fitted_grid, pld_points, te_points = create_surface_data(
        tis, tes, ntes, fitted_signal
    )

    return pld_grid, te_grid, fitted_grid, pld_points, te_points
    
    
    
    
    

def plot_voxel_fit_multite(pwi_data, tis, tes, ntes, m0_data, taus,
                           voxel_coords, output_path=None,
                           T1=1.3, T1a=1.65, T2=0.050, T2a=0.150,
                           texch=0.1, itt=0.2, lambd=0.9, alpha=0.68):

    from model_multi_te import deltaM_multite_model

    x, y, z = voxel_coords

    # Get signal and M0 for this voxel
    signal = pwi_data[x, y, z, :]
    m0 = m0_data[x, y, z]

    # Check for valid data
    if (np.any(np.isnan(signal)) or np.any(np.isinf(signal)) or
            np.all(signal == 0) or np.isnan(m0) or np.isinf(m0) or m0 <= 0):
        print(f"Invalid data at voxel ({x}, {y}, {z})")
        return None, None, None, None

    # Normalize signal
    signal_normalized = signal / (m0 * 5)
    M0a = m0 / (6000 * 0.9)

    # Perform LS fitting
    print(f"LS fitting voxel ({x}, {y}, {z})...")
    att_fit, cbf_fit, rmse = ls_fit_voxel_multite(
        tis, tes, ntes, signal_normalized, M0a, taus,
        T1=T1, T1a=T1a, T2=T2, T2a=T2a,
        texch=texch, itt=itt, lambd=lambd, alpha=alpha
    )

    if np.isnan(att_fit) or np.isnan(cbf_fit):
        print(f"LS fitting failed for voxel ({x}, {y}, {z})")
        return None, None, None, None

    # Perform Bayesian fitting
    print("Bayesian fitting...")
    bayes_config = create_multite_bayesian_config()
    bayes_config.update({
        "T1": T1, "T1a": T1a, "T2": T2, "T2a": T2a,
        "texch": texch, "itt": itt, "lambd": lambd, "alpha": alpha
    })

    fit = bayesian_fit_voxel_multite(
        tis, tes, ntes, signal_normalized, M0a, taus,
        bayes_config,
        att_ls_val=att_fit,
        cbf_ls_val=cbf_fit
    )

    bayes_results = None
    if fit is not None:
        try:
            df = fit.to_frame()
            att_mean = df['att'].mean()
            att_std = df['att'].std()
            cbf_mean = df['cbf'].mean()
            cbf_std = df['cbf'].std()
            print(f"Bayesian results - ATT: {att_mean:.3f}±{att_std:.3f}s, "
                  f"CBF: {cbf_mean:.1f}±{cbf_std:.1f} ml/min/100g")
            bayes_results = (att_mean, cbf_mean, att_std, cbf_std)
        except Exception as e:
            print(f"Error extracting Bayesian results: {e}")
    else:
        print("Bayesian fitting failed")

    # Generate fitted signals
    fitted_signal_ls = deltaM_multite_model(
        tis, tes, ntes, att_fit, cbf_fit, M0a, taus,
        t1=T1, t1b=T1a, t2=T2, t2b=T2a,
        texch=texch, itt=itt, lambd=lambd, alpha=alpha
    )

    fitted_signal_bayes = None
    if bayes_results is not None:
        fitted_signal_bayes = deltaM_multite_model(
            tis, tes, ntes, bayes_results[0], bayes_results[1], M0a, taus,
            t1=T1, t1b=T1a, t2=T2, t2b=T2a,
            texch=texch, itt=itt, lambd=lambd, alpha=alpha
        )

    # Create enhanced plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: All measurements
    measurement_indices = np.arange(len(signal_normalized))
    ax1.plot(measurement_indices, signal_normalized, 'bo-',
             linewidth=2, markersize=6, label='Measured data')
    ax1.plot(measurement_indices, fitted_signal_ls, 'r-',
             linewidth=2, label='LS fit')

    if fitted_signal_bayes is not None:
        ax1.plot(measurement_indices, fitted_signal_bayes, 'g--',
                 linewidth=2, label='Bayesian fit')

    ax1.set_xlabel('Measurement Index')
    ax1.set_ylabel('Normalised Signal')

    title = f'Multi-TE ASL Fit - Voxel ({x}, {y}, {z})\n'
    title += f'LS: ATT={att_fit:.3f}s, CBF={cbf_fit:.1f} ml/min/100g, RMSE={rmse:.6f}'
    if bayes_results is not None:
        title += f'\nBayes: ATT={bayes_results[0]:.3f}±{bayes_results[2]:.3f}s, '
        title += f'CBF={bayes_results[1]:.1f}±{bayes_results[3]:.1f} ml/min/100g'

    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Signal vs TI for different TEs
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(tes))))
    te_unique = np.unique(tes)

    for i, te_val in enumerate(te_unique):
        te_indices = []
        current_idx = 0
        for j, ti_val in enumerate(tis):
            for k in range(ntes[j]):
                if np.isclose(tes[current_idx], te_val):
                    te_indices.append(current_idx)
                current_idx += 1

        if te_indices:
            te_tis, te_signals, te_fitted_ls, te_fitted_bayes = [], [], [], []
            current_idx = 0
            for j, ti_val in enumerate(tis):
                for k in range(ntes[j]):
                    if current_idx in te_indices:
                        te_tis.append(ti_val)
                        te_signals.append(signal_normalized[current_idx])
                        te_fitted_ls.append(fitted_signal_ls[current_idx])
                        if fitted_signal_bayes is not None:
                            te_fitted_bayes.append(fitted_signal_bayes[current_idx])
                    current_idx += 1

            ax2.plot(te_tis, te_signals, 'o', color=colors[i],
                     markersize=6, label=f'TE = {te_val * 1000:.1f} ms (data)')
            ax2.plot(te_tis, te_fitted_ls, '-', color=colors[i],
                     linewidth=2, alpha=0.8)

            if fitted_signal_bayes is not None:
                ax2.plot(te_tis, te_fitted_bayes, '--', color=colors[i],
                         linewidth=2, alpha=0.6)

    ax2.set_xlabel('Inversion Time (s)')
    ax2.set_ylabel('Normalised Signal')
    ax2.set_title('Signal vs TI for different TEs')

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='black', linestyle='',
                              markersize=6, label='Measured data'),
                       Line2D([0], [0], color='black', linestyle='-',
                              linewidth=2, label='LS fit')]
    if fitted_signal_bayes is not None:
        legend_elements.append(Line2D([0], [0], color='black', linestyle='--',
                                      linewidth=2, label='Bayesian fit'))

    ax2.legend(handles=legend_elements)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    plt.show()

    # Print results
    print(f"\n=== Fit Results for Voxel ({x}, {y}, {z}) ===")
    print(f"LS: ATT = {att_fit:.3f}s, CBF = {cbf_fit:.1f} ml/min/100g, RMSE = {rmse:.6f}")
    if bayes_results is not None:
        print(f"Bayesian: ATT = {bayes_results[0]:.3f}±{bayes_results[2]:.3f}s, "
              f"CBF = {bayes_results[1]:.1f}±{bayes_results[3]:.1f} ml/min/100g")

    return att_fit, cbf_fit, rmse, bayes_results
    
    
    

def plot_voxel_fit_multite(pwi_data, tis, tes, ntes, m0_data, taus,
                           voxel_coords, output_path=None,
                           T1=1.3, T1a=1.65, T2=0.050, T2a=0.150,
                           texch=0.1, itt=0.2, lambd=0.9, alpha=0.68):

    from model_multi_te import deltaM_multite_model

    x, y, z = voxel_coords

    # Get signal and M0 for this voxel
    signal = pwi_data[x, y, z, :]
    m0 = m0_data[x, y, z]

    # Check for valid data
    if (np.any(np.isnan(signal)) or np.any(np.isinf(signal)) or
            np.all(signal == 0) or np.isnan(m0) or np.isinf(m0) or m0 <= 0):
        print(f"Invalid data at voxel ({x}, {y}, {z})")
        return None, None, None, None

    # Normalize signal
    signal_normalized = signal / (m0 * 5)
    M0a = m0 / (6000 * 0.9)

    # Perform LS fitting
    print(f"LS fitting voxel ({x}, {y}, {z})...")
    att_fit, cbf_fit, rmse = ls_fit_voxel_multite(
        tis, tes, ntes, signal_normalized, M0a, taus,
        T1=T1, T1a=T1a, T2=T2, T2a=T2a,
        texch=texch, itt=itt, lambd=lambd, alpha=alpha
    )

    if np.isnan(att_fit) or np.isnan(cbf_fit):
        print(f"LS fitting failed for voxel ({x}, {y}, {z})")
        return None, None, None, None

    # Perform Bayesian fitting
    print("Bayesian fitting...")
    bayes_config = create_multite_bayesian_config()
    bayes_config.update({
        "T1": T1, "T1a": T1a, "T2": T2, "T2a": T2a,
        "texch": texch, "itt": itt, "lambd": lambd, "alpha": alpha
    })

    fit = bayesian_fit_voxel_multite(
        tis, tes, ntes, signal_normalized, M0a, taus,
        bayes_config,
        att_ls_val=att_fit,
        cbf_ls_val=cbf_fit
    )

    bayes_results = None
    if fit is not None:
        try:
            df = fit.to_frame()
            att_mean = df['att'].mean()
            att_std = df['att'].std()
            cbf_mean = df['cbf'].mean()
            cbf_std = df['cbf'].std()
            print(f"Bayesian results - ATT: {att_mean:.3f}±{att_std:.3f}s, "
                  f"CBF: {cbf_mean:.1f}±{cbf_std:.1f} ml/min/100g")
            bayes_results = (att_mean, cbf_mean, att_std, cbf_std)
        except Exception as e:
            print(f"Error extracting Bayesian results: {e}")
    else:
        print("Bayesian fitting failed")

    # Generate fitted signals
    fitted_signal_ls = deltaM_multite_model(
        tis, tes, ntes, att_fit, cbf_fit, M0a, taus,
        t1=T1, t1b=T1a, t2=T2, t2b=T2a,
        texch=texch, itt=itt, lambd=lambd, alpha=alpha
    )

    fitted_signal_bayes = None
    if bayes_results is not None:
        fitted_signal_bayes = deltaM_multite_model(
            tis, tes, ntes, bayes_results[0], bayes_results[1], M0a, taus,
            t1=T1, t1b=T1a, t2=T2, t2b=T2a,
            texch=texch, itt=itt, lambd=lambd, alpha=alpha
        )

    # Create enhanced plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: All measurements
    measurement_indices = np.arange(len(signal_normalized))
    ax1.plot(measurement_indices, signal_normalized, 'bo-',
             linewidth=2, markersize=6, label='Measured data')
    ax1.plot(measurement_indices, fitted_signal_ls, 'r-',
             linewidth=2, label='LS fit')

    if fitted_signal_bayes is not None:
        ax1.plot(measurement_indices, fitted_signal_bayes, 'g--',
                 linewidth=2, label='Bayesian fit')

    ax1.set_xlabel('Measurement Index')
    ax1.set_ylabel('Normalised Signal')

    title = f'Multi-TE ASL Fit - Voxel ({x}, {y}, {z})\n'
    title += f'LS: ATT={att_fit:.3f}s, CBF={cbf_fit:.1f} ml/min/100g, RMSE={rmse:.6f}'
    if bayes_results is not None:
        title += f'\nBayes: ATT={bayes_results[0]:.3f}±{bayes_results[2]:.3f}s, '
        title += f'CBF={bayes_results[1]:.1f}±{bayes_results[3]:.1f} ml/min/100g'

    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Signal vs TI for different TEs
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(tes))))
    te_unique = np.unique(tes)

    for i, te_val in enumerate(te_unique):
        te_indices = []
        current_idx = 0
        for j, ti_val in enumerate(tis):
            for k in range(ntes[j]):
                if np.isclose(tes[current_idx], te_val):
                    te_indices.append(current_idx)
                current_idx += 1

        if te_indices:
            te_tis, te_signals, te_fitted_ls, te_fitted_bayes = [], [], [], []
            current_idx = 0
            for j, ti_val in enumerate(tis):
                for k in range(ntes[j]):
                    if current_idx in te_indices:
                        te_tis.append(ti_val)
                        te_signals.append(signal_normalized[current_idx])
                        te_fitted_ls.append(fitted_signal_ls[current_idx])
                        if fitted_signal_bayes is not None:
                            te_fitted_bayes.append(fitted_signal_bayes[current_idx])
                    current_idx += 1

            ax2.plot(te_tis, te_signals, 'o', color=colors[i],
                     markersize=6, label=f'TE = {te_val * 1000:.1f} ms (data)')
            ax2.plot(te_tis, te_fitted_ls, '-', color=colors[i],
                     linewidth=2, alpha=0.8)

            if fitted_signal_bayes is not None:
                ax2.plot(te_tis, te_fitted_bayes, '--', color=colors[i],
                         linewidth=2, alpha=0.6)

    ax2.set_xlabel('Inversion Time (s)')
    ax2.set_ylabel('Normalised Signal')
    ax2.set_title('Signal vs TI for different TEs')

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='black', linestyle='',
                              markersize=6, label='Measured data'),
                       Line2D([0], [0], color='black', linestyle='-',
                              linewidth=2, label='LS fit')]
    if fitted_signal_bayes is not None:
        legend_elements.append(Line2D([0], [0], color='black', linestyle='--',
                                      linewidth=2, label='Bayesian fit'))

    ax2.legend(handles=legend_elements)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    plt.show()

    # Print results
    print(f"\n=== Fit Results for Voxel ({x}, {y}, {z}) ===")
    print(f"LS: ATT = {att_fit:.3f}s, CBF = {cbf_fit:.1f} ml/min/100g, RMSE = {rmse:.6f}")
    if bayes_results is not None:
        print(f"Bayesian: ATT = {bayes_results[0]:.3f}±{bayes_results[2]:.3f}s, "
              f"CBF = {bayes_results[1]:.1f}±{bayes_results[3]:.1f} ml/min/100g")

    return att_fit, cbf_fit, rmse, bayes_results
    
            

"""