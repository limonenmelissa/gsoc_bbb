import os
import numpy as np
import csv

def save_results_to_csv(results, data_dir):
	"""
	Save fitting results to CSV file
	"""

	csv_path = os.path.join(data_dir, "results_flexible_with_ls_priors.csv")

	fieldnames = ['x', 'y', 'z', 'att_mean', 'att_std', 'cbf_mean', 'cbf_std',
				  'att_lm', 'cbf_lm', 'att_diff', 'cbf_diff',
				  'T1_blood', 'T1_tissue', 'lambda', 'alpha']

	# Add fitted parameter fields
	if results['individual_results']:
		first_result = results['individual_results'][0]
		fitted_params = first_result.get('fitted_parameters', {})
		for param_name in fitted_params:
			fieldnames.extend([param_name, param_name.replace('_mean', '_std')])

	with open(csv_path, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		for result in results['individual_results']:
			x, y, z = result['voxel']
			att_lm = result['att_lm']
			cbf_lm = result['cbf_lm']

			# Calculate differences from LS
			att_diff = result['att_mean'] - att_lm if np.isfinite(att_lm) else np.nan
			cbf_diff = result['cbf_mean'] - cbf_lm if np.isfinite(cbf_lm) else np.nan

			row = {
				'x': x, 'y': y, 'z': z,
				'att_mean': result['att_mean'],
				'att_std': result['att_std'],
				'cbf_mean': result['cbf_mean'],
				'cbf_std': result['cbf_std'],
				'att_lm': att_lm,
				'cbf_lm': cbf_lm,
				'att_diff': att_diff,
				'cbf_diff': cbf_diff,
				'T1_blood': result['parameters']['T1_blood'],
				'T1_tissue': result['parameters']['T1_tissue'],
				'lambda': result['parameters']['lambda'],
				'alpha': result['parameters']['alpha']
			}

			# Add fitted parameter values
			fitted_params = result.get('fitted_parameters', {})
			for param_name, value in fitted_params.items():
				row[param_name] = value

			writer.writerow(row)

	print(f"Saved individual results to {csv_path}")

def save_ls_results_summary_csv(att_map_lm, cbf_map_lm, data_dir):
	"""
	Save a summary CSV with only valid LS results (excluding NaN values)
	"""

	csv_path = os.path.join(data_dir, "LS_results_valid_only.csv")

	# Find all valid (non-NaN) voxels
	valid_mask = (~np.isnan(att_map_lm)) & (~np.isnan(cbf_map_lm))
	valid_coords = np.where(valid_mask)

	fieldnames = ['x', 'y', 'z', 'att_ls', 'cbf_ls']

	print(f"Saving {len(valid_coords[0])} valid LS results to summary CSV...")

	with open(csv_path, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		for i in range(len(valid_coords[0])):
			x, y, z = valid_coords[0][i], valid_coords[1][i], valid_coords[2][i]

			row = {
				'x': x,
				'y': y,
				'z': z,
				'att_ls': att_map_lm[x, y, z],
				'cbf_ls': cbf_map_lm[x, y, z]
			}

			writer.writerow(row)

	print(f"Summary CSV file saved: {csv_path}")
	return csv_path

def save_ls_results_to_csv(att_map_lm, cbf_map_lm, pwi_data, m0_data, data_dir,
						   additional_info=None):
	"""
	Save Least Squares fitting results for all voxels to CSV file

	Parameters:
	- att_map_lm: 3D array with ATT values from LS fitting
	- cbf_map_lm: 3D array with CBF values from LS fitting
	- pwi_data: 4D PWI data (for validation checks)
	- m0_data: 3D M0 data (for validation checks)
	"""

	csv_path = os.path.join(data_dir, "LS_results_all_voxels.csv")

	# Define CSV headers
	fieldnames = ['x', 'y', 'z', 'att_ls', 'cbf_ls', 'valid_fit',
				  'has_valid_signal', 'has_valid_m0', 'm0_value']

	# Add additional info fields if provided
	if additional_info:
		for key in additional_info.keys():
			if key not in fieldnames:
				fieldnames.append(key)

	shape = att_map_lm.shape
	total_voxels = shape[0] * shape[1] * shape[2]
	valid_fits = 0
	invalid_fits = 0

	print(f"Saving LS results for all {total_voxels} voxels to CSV...")

	with open(csv_path, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		for x in range(shape[0]):
			for y in range(shape[1]):
				for z in range(shape[2]):

					# Get LS results
					att_val = att_map_lm[x, y, z]
					cbf_val = cbf_map_lm[x, y, z]

					# Check if fit is valid (not NaN)
					valid_fit = not (np.isnan(att_val) or np.isnan(cbf_val))
					if valid_fit:
						valid_fits += 1
					else:
						invalid_fits += 1

					# Check signal validity
					signal = pwi_data[x, y, z, :]
					has_valid_signal = not (np.any(np.isnan(signal)) or
											np.any(np.isinf(signal)) or
											np.all(signal == 0))

					# Check M0 validity
					m0_val = m0_data[x, y, z]
					has_valid_m0 = not (np.isnan(m0_val) or np.isinf(m0_val) or m0_val <= 0)

					# Prepare row data
					row = {
						'x': x,
						'y': y,
						'z': z,
						'att_ls': att_val if not np.isnan(att_val) else '',
						'cbf_ls': cbf_val if not np.isnan(cbf_val) else '',
						'valid_fit': valid_fit,
						'has_valid_signal': has_valid_signal,
						'has_valid_m0': has_valid_m0,
						'm0_value': m0_val if not np.isnan(m0_val) else ''
					}

					# Add additional info if provided
					if additional_info:
						for key, value in additional_info.items():
							if hasattr(value, 'shape') and len(value.shape) == 3:
								# If it's a 3D array, get the value for this voxel
								row[key] = value[x, y, z]
							else:
								# If it's a scalar or other value, use it directly
								row[key] = value

					writer.writerow(row)

	print(f"CSV file saved: {csv_path}")
	print(f"Total voxels: {total_voxels}")
	print(f"Valid fits: {valid_fits} ({100 * valid_fits / total_voxels:.1f}%)")
	print(f"Invalid fits: {invalid_fits} ({100 * invalid_fits / total_voxels:.1f}%)")

	return csv_path



def save_ls_results_summary_csv_ext(att_map_lm, cbf_map_lm, abv_map_lm, att_a_map_lm, data_dir):
	"""
	Save a summary CSV with only valid extended LS results (excluding NaN values)
	"""
	csv_path = os.path.join(data_dir, "LS_results_ext_valid_only.csv")

	# Find all valid (non-NaN) voxels
	valid_mask = (~np.isnan(att_map_lm)) & (~np.isnan(cbf_map_lm)) & (~np.isnan(abv_map_lm)) & (~np.isnan(att_a_map_lm))
	valid_coords = np.where(valid_mask)

	fieldnames = ['x', 'y', 'z', 'att_ls', 'cbf_ls', 'abv_ls', 'att_a_ls']

	print(f"Saving {len(valid_coords[0])} valid extended LS results to summary CSV...")

	with open(csv_path, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		for i in range(len(valid_coords[0])):
			x, y, z = valid_coords[0][i], valid_coords[1][i], valid_coords[2][i]

			row = {
				'x': x,
				'y': y,
				'z': z,
				'att_ls': att_map_lm[x, y, z],
				'cbf_ls': cbf_map_lm[x, y, z],
				'abv_ls': abv_map_lm[x, y, z],
				'att_a_ls': att_a_map_lm[x, y, z]
			}

			writer.writerow(row)

	print(f"Extended summary CSV file saved: {csv_path}")
	return csv_path


def save_ls_results_to_csv_ext(att_map_lm, cbf_map_lm, abv_map_lm, att_a_map_lm,
							   pwi_data, m0_data, data_dir, additional_info=None):
	"""
	Save extended Least Squares fitting results for all voxels to CSV file

	Parameters:
	- att_map_lm: 3D array with ATT values from LS fitting
	- cbf_map_lm: 3D array with CBF values from LS fitting
	- abv_map_lm: 3D array with ABV values from LS fitting
	- att_a_map_lm: 3D array with ATT_A values from LS fitting
	- pwi_data: 4D PWI data (for validation checks)
	- m0_data: 3D M0 data (for validation checks)
	"""
	csv_path = os.path.join(data_dir, "LS_results_ext_all_voxels.csv")

	# Define CSV headers
	fieldnames = ['x', 'y', 'z', 'att_ls', 'cbf_ls', 'abv_ls', 'att_a_ls',
				  'valid_fit', 'has_valid_signal', 'has_valid_m0', 'm0_value']

	# Add additional info fields if provided
	if additional_info:
		for key in additional_info.keys():
			if key not in fieldnames:
				fieldnames.append(key)

	shape = att_map_lm.shape
	total_voxels = shape[0] * shape[1] * shape[2]
	valid_fits = 0
	invalid_fits = 0

	print(f"Saving extended LS results for all {total_voxels} voxels to CSV...")

	with open(csv_path, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		for x in range(shape[0]):
			for y in range(shape[1]):
				for z in range(shape[2]):

					# Get LS results
					att_val = att_map_lm[x, y, z]
					cbf_val = cbf_map_lm[x, y, z]
					abv_val = abv_map_lm[x, y, z] if abv_map_lm is not None else np.nan
					att_a_val = att_a_map_lm[x, y, z] if att_a_map_lm is not None else np.nan

					# Check if fit is valid (not NaN)
					valid_fit = not (np.isnan(att_val) or np.isnan(cbf_val) or
									 np.isnan(abv_val) or np.isnan(att_a_val))
					if valid_fit:
						valid_fits += 1
					else:
						invalid_fits += 1

					# Check signal validity
					signal = pwi_data[x, y, z, :]
					has_valid_signal = not (np.any(np.isnan(signal)) or
											np.any(np.isinf(signal)) or
											np.all(signal == 0))

					# Check M0 validity
					m0_val = m0_data[x, y, z]
					has_valid_m0 = not (np.isnan(m0_val) or np.isinf(m0_val) or m0_val <= 0)

					# Prepare row data
					row = {
						'x': x,
						'y': y,
						'z': z,
						'att_ls': att_val if not np.isnan(att_val) else '',
						'cbf_ls': cbf_val if not np.isnan(cbf_val) else '',
						'abv_ls': abv_val if not np.isnan(abv_val) else '',
						'att_a_ls': att_a_val if not np.isnan(att_a_val) else '',
						'valid_fit': valid_fit,
						'has_valid_signal': has_valid_signal,
						'has_valid_m0': has_valid_m0,
						'm0_value': m0_val if not np.isnan(m0_val) else ''
					}

					# Add additional info if provided
					if additional_info:
						for key, value in additional_info.items():
							if hasattr(value, 'shape') and len(value.shape) == 3:
								# If it's a 3D array, get the value for this voxel
								row[key] = value[x, y, z]
							else:
								# If it's a scalar or other value, use it directly
								row[key] = value

					writer.writerow(row)

	print(f"Extended CSV file saved: {csv_path}")
	print(f"Total voxels: {total_voxels}")
	print(f"Valid fits: {valid_fits} ({100 * valid_fits / total_voxels:.1f}%)")
	print(f"Invalid fits: {invalid_fits} ({100 * invalid_fits / total_voxels:.1f}%)")

	return csv_path
