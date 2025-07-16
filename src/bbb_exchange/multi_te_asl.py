import os
import numpy as np
import nibabel as nib
from data_handling import load_nifti_file, load_json_metadata
from fitting_multi_te import fit_volume_2params, fit_volume_extended
import matplotlib.pyplot as plt


def save_nifti(data, ref_img, out_path):
	"""
	Save data as NIfTI file
	"""
	img = nib.Nifti1Image(data, affine=ref_img.affine, header=ref_img.header)
	nib.save(img, out_path)
	print(f"Saved {out_path}")


def create_te_ti_arrays(echo_times, plds, te_per_ti):
	"""
	Create arrays for multi-TE fitting

	Parameters:
	- echo_times: all echo times from json file
	- plds: all post-labeling delays
	- te_per_ti: number of echo times per inversion time

	Returns:
	- tis: inversion times
	- tes: all echo times in acquisition order
	- ntes: number of TEs per TI
	"""

	# Get TIs and sort them
	tis = np.unique(plds)
	tis = np.sort(tis)

	# Create the full TE array
	tes = np.tile(echo_times[:te_per_ti], len(tis))

	# Number of TEs per TI
	ntes = np.full(len(tis), te_per_ti)

	return tis, tes, ntes



def multi_te_asl():

	print("Reading data for multi_te ASL ...")
	# Read data
	script_dir = os.path.dirname(os.path.abspath(__file__))
	data_dir = os.path.join(script_dir, "..", "data", "multite")

	pwi_nii_path = os.path.join(data_dir, "PWI4D.nii")
	pwi_json_path = os.path.join(data_dir, "PWI4D.json")
	m0_nii_path = os.path.join(data_dir, "M0.nii.gz")
	m0_json_path = os.path.join(data_dir, "M0.json")

	print("Loading data...")

	# Load data
	pwi_img, pwi_data_full = load_nifti_file(pwi_nii_path)
	m0_img, m0_data_full = load_nifti_file(m0_nii_path)
	pwi_meta = load_json_metadata(pwi_json_path)
	m0_meta = load_json_metadata(m0_json_path)

	# Extract parameters from file
	echo_times = np.array(pwi_meta["EchoTime"])
	plds = np.array(pwi_meta["PostLabelingDelay"])

	tau_raw = pwi_meta.get("LabelingDuration", 1.8)
	if isinstance(tau_raw, list):
		taus = np.array(tau_raw)
	else:
		taus = tau_raw

	if m0_data_full.ndim == 4 and m0_data_full.shape[3] == 1:
		m0_data = m0_data_full[:, :, :, 0]
	else:
		m0_data = m0_data_full

	te_per_ti = len(np.unique(echo_times))

	print(f"- Unique echo times: {len(np.unique(echo_times))}")
	print(f"- Unique inversion times: {len(np.unique(plds))}")
	print(f"- Total volumes: {pwi_data_full.shape[3]}")

	# Create TI and TE arrays for the fitting
	tis, tes, ntes = create_te_ti_arrays(echo_times, plds, te_per_ti)

	print(f"- TIs: {tis}")
	print(f"- TEs: {tes[:te_per_ti]} (repeated for each TI)")
	print(f"- Number of TEs per TI: {ntes[0]}")
	print(f"- Tau: {taus}")

	# Use above data for multi-TE fitting
	pwi_data = pwi_data_full

	print(f"PWI data shape: {pwi_data.shape}")
	print(f"M0 data shape: {m0_data.shape}")

	# Handle NaN values by setting them zo zero
	nan_count = np.sum(np.isnan(pwi_data))
	total_elements = np.prod(pwi_data.shape)
	# How large is the proportion of NaN data?
	print(f"NaN values in PWI data: {nan_count}/{total_elements} ({100 * nan_count / total_elements:.2f}%)")

	if nan_count > 0:
		print("WARNING: PWI data contains NaN values!")
		pwi_data = np.nan_to_num(pwi_data, nan=0.0)
		print("Replaced NaN values with 0")

	# Data range check
	print(f"PWI data range: [{np.min(pwi_data):.6f}, {np.max(pwi_data):.6f}]")
	print(f"M0 data range: [{np.min(m0_data):.6f}, {np.max(m0_data):.6f}]")

	# 2-parameter fitting (ATT, CBF)
	print("\nRunning 2-parameter multi-TE fitting (ATT, CBF)...")
	att_map_2params, cbf_map_basic, att_std_map, cbf_std_map = fit_volume_2params(
		pwi_data, tis, tes, ntes, m0_data, taus
	)

	# Save 2 parameter fitting results
	print("Saving 2 parameter fitting results...")
	save_nifti(att_map_2params, pwi_img, os.path.join(data_dir, "ATT_map_2params.nii.gz"))
	save_nifti(cbf_map_basic, pwi_img, os.path.join(data_dir, "CBF_map_2params.nii.gz"))
	save_nifti(att_std_map, pwi_img, os.path.join(data_dir, "ATT_std_map.nii.gz"))
	save_nifti(cbf_std_map, pwi_img, os.path.join(data_dir, "CBF_std_map.nii.gz"))

	# Extended 4-parameter fitting (ATT, CBF, texch, itt)
	print("\nRunning extended multi-TE fitting (ATT, CBF, texch, itt)...")
	try:
		(att_map_ext, cbf_map_ext, texch_map, itt_map,
		 att_std_ext, cbf_std_ext, texch_std_map, itt_std_map) = fit_volume_extended(
			pwi_data, tis, tes, ntes, m0_data, taus
		)

		# Save extended fitting results
		print("Saving extended fitting results...")
		save_nifti(att_map_ext, pwi_img, os.path.join(data_dir, "ATT_map_extended.nii.gz"))
		save_nifti(cbf_map_ext, pwi_img, os.path.join(data_dir, "CBF_map_extended.nii.gz"))
		save_nifti(texch_map, pwi_img, os.path.join(data_dir, "texch_map.nii.gz"))
		save_nifti(itt_map, pwi_img, os.path.join(data_dir, "itt_map.nii.gz"))
		save_nifti(att_std_ext, pwi_img, os.path.join(data_dir, "ATT_std_extended.nii.gz"))
		save_nifti(cbf_std_ext, pwi_img, os.path.join(data_dir, "CBF_std_extended.nii.gz"))
		save_nifti(texch_std_map, pwi_img, os.path.join(data_dir, "texch_std_map.nii.gz"))
		save_nifti(itt_std_map, pwi_img, os.path.join(data_dir, "itt_std_map.nii.gz"))

	except Exception as e:
		print(f"Extended fitting failed: {e}")
		print("Continuing with basic fitting results only...")


	# Print summary
	print("\nSummary:")
	print("2 Parameter Fitting:")
	valid_att = att_map_2params[~np.isnan(att_map_2params)]
	valid_cbf = cbf_map_basic[~np.isnan(cbf_map_basic)]

	if len(valid_att) > 0:
		print(f"  ATT: {np.mean(valid_att):.3f} ± {np.std(valid_att):.3f} s")
		print(f"  CBF: {np.mean(valid_cbf):.1f} ± {np.std(valid_cbf):.1f} ml/min/100g")

	print("\nMulti-TE ASL processing completed!")


if __name__ == "__main__":
	multi_te_asl()