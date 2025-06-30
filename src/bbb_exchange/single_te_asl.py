import os
import numpy as np
import nibabel as nib
from data_handling import load_nifti_file, load_json_metadata
from fitting import fit_voxel, fit_volume
from voxelwise_model import deltaM_model


def save_nifti(data, ref_img, out_path):
	img = nib.Nifti1Image(data, affine=ref_img.affine, header=ref_img.header)
	nib.save(img, out_path)
	print(f"Saved {out_path}")


def asl():
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

	chosen_te = 0.01302
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

	# check for NaN in PWI4D.nii
	nan_count = np.sum(np.isnan(pwi_data))
	total_elements = np.prod(pwi_data.shape)
	print(f"NaN values in PWI data: {nan_count}/{total_elements} ({100 * nan_count / total_elements:.2f}%)")

	if nan_count > 0:
		print("WARNING: PWI data contains NaN values!")
		# substitute NaN by zeros
		pwi_data = np.nan_to_num(pwi_data, nan=0.0)
		print("Replaced NaN values with 0")

	# Check data range
	print(f"PWI data range after NaN handling: [{np.min(pwi_data):.6f}, {np.max(pwi_data):.6f}]")
	print(f"M0 data range: [{np.min(m0_data):.6f}, {np.max(m0_data):.6f}]")

	print(f"Using {len(indices)} volumes with TE = {chosen_te}")

	# Standard LM fitting
	print("Running LM fitting ...")
	att_map_lm, cbf_map_lm = fit_volume(pwi_data, t, m0_data, tau)

	print("Saving LM results ...")
	save_nifti(att_map_lm, pwi_img, os.path.join(data_dir, "ATT_map_LM.nii.gz"))
	save_nifti(cbf_map_lm, pwi_img, os.path.join(data_dir, "CBF_map_LM.nii.gz"))

	print("Done!")


if __name__ == "__main__":
	asl()