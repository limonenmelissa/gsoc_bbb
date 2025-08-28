import os
import numpy as np
from data_handling import load_nifti_file, load_json_metadata, save_nifti
from fitting_multi_te import ls_fit_volume_multite, prepare_multite_data
from fitting_multi_te import bayesian_fit_volume_multite, create_multite_bayesian_config
import json
with open("config.json", "r") as file:
		config = json.load(file)

"""
Multi-TE ASL Processing Script with Bayesian Fitting

This script performs Multi-TE ASL analysis with the following steps:

1. Load configuration and data (PWI 4D, M0, metadata)
2. Prepare Multi-TE data structure (TIs, TEs, etc.)
3. Analyse data for NaN values
4. Perform Least Squares fitting using Multi-TE model
5. Perform Bayesian fitting using Multi-TE model
6. Save results and generate visualization for a test voxel

The Multi-TE model incorporates:
- Multiple echo times per inversion time
- Three-compartment model (blood and tissue compartments + extravascular)
- T1 and T2 decay effects
- Exchange between compartments

Output:
- NIfTI images of fitted CBF, ATT, and RMSE/uncertainty maps
- Visualization of fit quality for a specified voxel
- CSV file with results summary (optional)
"""


def create_multite_config():
    """
    Create Multi-TE specific configuration from config.json
    """
    multite_config = {
        # Multi-TE model parameters from config.json
        'T1': config['physiological']['T1'],  # T1 of tissue [s]
        'T1a': config['physiological']['T1a'],  # T1 of blood [s]
        'T2': config['physiological']['T2'],  # T2 of tissue [s]
        'T2a': config['physiological']['T2a'],  # T2 of blood [s]
        'texch': config['physiological']['texch'],  # Exchange time [s]
        'itt': config['physiological']['itt'],  # Intra-voxel transit time [s]
        'lambd': config['physiological']['lambd'],  # Blood-tissue partition coefficient
        'alpha': config['physiological']['a'],  # Labeling efficiency

        # Test voxel coordinates for visualization from config.json
        'test_voxel': config['multi_te']['test_voxel'],  # [x, y, z] coordinates

        # Processing options from config.json
        'run_ls_fitting': config['multi_te']['run_ls_fitting'],
        'run_bayesian_fitting': config['multi_te']['run_bayesian_fitting'],
        'max_bayesian_voxels': config['multi_te']['max_bayesian_voxels'],  # Limit to debug Bayesian fitting

        # Output options from config.json
        'save_test_plot': config['multi_te']['save_test_plot'],
        'save_csv': config['multi_te']['save_csv']
    }
    return multite_config


def asl_multite():
    """
    Multi-TE ASL processing function with both LS and Bayesian fitting
    """

    print("=== Multi-TE ASL Processing with Bayesian Fitting ===")

    # 1. === Load configuration ===
    print("\n=== Loading Configuration ===")
    multite_config = create_multite_config()

    # Print parameters
    print(f"T1 = {multite_config['T1']}s, T1a = {multite_config['T1a']}s")
    print(f"T2 = {multite_config['T2']}s, T2a = {multite_config['T2a']}s")
    print(f"Exchange time = {multite_config['texch']}s")
    print(f"Intra-voxel transit time = {multite_config['itt']}s")
    print(f"Lambda = {multite_config['lambd']}, Alpha = {multite_config['alpha']}")
    print(f"LS fitting: {multite_config['run_ls_fitting']}")
    print(f"Bayesian fitting: {multite_config['run_bayesian_fitting']}")

    # 2. === Load data ===
    print("\n=== Loading Data ===")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data", "multite")

    pwi_path = os.path.join(data_dir, "PWI4D.nii")
    pwi_json_path = os.path.join(data_dir, "PWI4D.json")
    m0_path = os.path.join(data_dir, "M0.nii.gz")
    m0_json_path = os.path.join(data_dir, "M0.json")

    print("Loading PWI and M0 data...")
    pwi_img, pwi_data_full = load_nifti_file(pwi_path)
    m0_img, m0_data_full = load_nifti_file(m0_path)
    pwi_meta = load_json_metadata(pwi_json_path)
    m0_meta = load_json_metadata(m0_json_path)

    # Handle M0 data dimensions
    if m0_data_full.ndim == 4 and m0_data_full.shape[3] == 1:
        m0_data = m0_data_full[:, :, :, 0]
    else:
        m0_data = m0_data_full

    print(f"PWI shape: {pwi_data_full.shape}, M0 shape: {m0_data.shape}")

    # 3. === Prepare Multi-TE data structure ===
    print("\n=== Preparing Multi-TE Data Structure ===")
    tis, tes, ntes, taus = prepare_multite_data(pwi_meta)

    # Verify data consistency
    expected_measurements = np.sum(ntes)
    actual_measurements = pwi_data_full.shape[3]

    if expected_measurements != actual_measurements:
        raise ValueError(f"Data inconsistency: expected {expected_measurements} measurements, "
                         f"got {actual_measurements}")

    print(f"Data consistency check passed: {actual_measurements} measurements")

    # 4. === Analyse data for NaN values ===
    print("\n=== Data Quality Analysis ===")
    nan_count = np.sum(np.isnan(pwi_data_full))
    total_elements = np.prod(pwi_data_full.shape)
    print(f"NaN values in PWI data: {nan_count}/{total_elements} ({100 * nan_count / total_elements:.2f}%)")

    m0_nan_count = np.sum(np.isnan(m0_data))
    m0_total_elements = np.prod(m0_data.shape)
    print(f"NaN values in M0 data: {m0_nan_count}/{m0_total_elements} ({100 * m0_nan_count / m0_total_elements:.2f}%)")

    # Check data range
    pwi_finite = pwi_data_full[np.isfinite(pwi_data_full)]
    m0_finite = m0_data[np.isfinite(m0_data)]

    if len(pwi_finite) > 0:
        print(f"PWI data range: [{np.min(pwi_finite):.6f}, {np.max(pwi_finite):.6f}]")
    if len(m0_finite) > 0:
        print(f"M0 data range: [{np.min(m0_finite):.6f}, {np.max(m0_finite):.6f}]")

    # Initialise result storage
    results = {}

    # 5. === Least Squares fitting ===
    att_map_lm, cbf_map_lm, rmse_map_lm = None, None, None

    if multite_config['run_ls_fitting']:
        print("\n=== Running Multi-TE Least Squares Fitting ===")

        att_map_lm, cbf_map_lm, rmse_map_lm = ls_fit_volume_multite(
            pwi_data_full, tis, tes, ntes, m0_data, taus,
            T1=multite_config['T1'],
            T1a=multite_config['T1a'],
            T2=multite_config['T2'],
            T2a=multite_config['T2a'],
            texch=multite_config['texch'],
            itt=multite_config['itt'],
            lambd=multite_config['lambd'],
            alpha=multite_config['alpha']
        )

        # Save LS results
        print("\n=== Saving LS Results ===")
        save_nifti(att_map_lm, pwi_img, os.path.join(data_dir, "ATT_map_MultiTE_LS.nii.gz"))
        save_nifti(cbf_map_lm, pwi_img, os.path.join(data_dir, "CBF_map_MultiTE_LS.nii.gz"))
        save_nifti(rmse_map_lm, pwi_img, os.path.join(data_dir, "RMSE_map_MultiTE_LS.nii.gz"))



        results['ls'] = {
            'att_map': att_map_lm,
            'cbf_map': cbf_map_lm,
            'rmse_map': rmse_map_lm
        }

    # 6. === Bayesian fitting ===
    bayesian_results = None

    if multite_config['run_bayesian_fitting']:
        print("\n=== Running Multi-TE Bayesian Fitting ===")

        # Create Bayesian configuration
        bayesian_config = create_multite_bayesian_config()
        bayesian_config.update(multite_config)
        bayesian_config['max_voxels'] = multite_config.get('max_bayesian_voxels')

        # Print Bayesian configuration
        print(f"Bayesian configuration:")
        print(f"  Max voxels: {bayesian_config['max_voxels']}")
        print(f"  Use LS priors: ATT={bayesian_config['att_prior_from_ls']}, CBF={bayesian_config['cbf_prior_from_ls']}")

        # Run Bayesian fitting
        bayesian_results = bayesian_fit_volume_multite(
            pwi_data_full, tis, tes, ntes, m0_data, taus,
            bayesian_config,
            att_ls_map=att_map_lm,
            cbf_ls_map=cbf_map_lm,
            max_voxels=bayesian_config['max_voxels']
        )

        if bayesian_results['successful_fits'] > 0:
            # Save Bayesian results
            print("\n=== Saving Bayesian Results ===")
            save_nifti(bayesian_results['att_map'], pwi_img,
                       os.path.join(data_dir, "ATT_map_MultiTE_Bayesian.nii.gz"))
            save_nifti(bayesian_results['cbf_map'], pwi_img,
                       os.path.join(data_dir, "CBF_map_MultiTE_Bayesian.nii.gz"))
            save_nifti(bayesian_results['att_std_map'], pwi_img,
                       os.path.join(data_dir, "ATT_std_map_MultiTE_Bayesian.nii.gz"))
            save_nifti(bayesian_results['cbf_std_map'], pwi_img,
                       os.path.join(data_dir, "CBF_std_map_MultiTE_Bayesian.nii.gz"))

            # Save fitted parameter maps if available
            param_names = ['T1', 'T1a', 'T2', 'T2a', 'texch', 'itt', 'lambd']
            for param_name in param_names:
                fitted_key = f'{param_name}_fitted_map'
                std_key = f'{param_name}_fitted_std_map'
                if fitted_key in bayesian_results:
                    save_nifti(bayesian_results[fitted_key], pwi_img,
                               os.path.join(data_dir, f"{param_name}_fitted_MultiTE_Bayesian.nii.gz"))
                    save_nifti(bayesian_results[std_key], pwi_img,
                               os.path.join(data_dir, f"{param_name}_std_MultiTE_Bayesian.nii.gz"))


            results['bayesian'] = bayesian_results
        else:
            print("No successful Bayesian fits obtained!")

    # 7. === Test voxel visualization ===
    print("\n=== Test Voxel Visualization ===")
    test_voxel = multite_config['test_voxel']
    print(f"Analyzing test voxel at coordinates: {test_voxel}")

    # Check if coordinates are valid
    shape = pwi_data_full.shape[:3]
    if (test_voxel[0] >= shape[0] or test_voxel[1] >= shape[1] or
            test_voxel[2] >= shape[2] or any(coord < 0 for coord in test_voxel)):
        print(f"Warning: Test voxel coordinates {test_voxel} are outside data bounds {shape}")
        print("Using center voxel instead...")
        test_voxel = [shape[0] // 2, shape[1] // 2, shape[2] // 2]

    # Plot fit for test voxel
    plot_path = os.path.join(data_dir, "test_voxel_fit_MultiTE.png") if multite_config['save_test_plot'] else None


    # 8. === Results comparison ===
    if multite_config['run_ls_fitting'] and multite_config['run_bayesian_fitting'] and bayesian_results[
        'successful_fits'] > 0:
        print("\n=== LS vs Bayesian Comparison ===")

        # Compare ATT
        att_ls_finite = att_map_lm[np.isfinite(att_map_lm)]
        att_bayes_finite = bayesian_results['att_map'][np.isfinite(bayesian_results['att_map'])]

        if len(att_ls_finite) > 0 and len(att_bayes_finite) > 0:
            print(f"ATT - LS: {np.mean(att_ls_finite):.3f}±{np.std(att_ls_finite):.3f}s")
            print(f"ATT - Bayesian: {np.mean(att_bayes_finite):.3f}±{np.std(att_bayes_finite):.3f}s")

        # Compare CBF
        cbf_ls_finite = cbf_map_lm[np.isfinite(cbf_map_lm)]
        cbf_bayes_finite = bayesian_results['cbf_map'][np.isfinite(bayesian_results['cbf_map'])]

        if len(cbf_ls_finite) > 0 and len(cbf_bayes_finite) > 0:
            print(f"CBF - LS: {np.mean(cbf_ls_finite):.1f}±{np.std(cbf_ls_finite):.1f} ml/min/100g")
            print(f"CBF - Bayesian: {np.mean(cbf_bayes_finite):.1f}±{np.std(cbf_bayes_finite):.1f} ml/min/100g")

    # 9. === Summary ===
    print("\n=== Processing Summary ===")

    if multite_config['run_ls_fitting']:
        valid_ls_voxels = np.sum(np.isfinite(att_map_lm))
        total_voxels = np.prod(shape)
        print(f"LS fitting: {valid_ls_voxels}/{total_voxels} voxels ({100 * valid_ls_voxels / total_voxels:.1f}%)")

    if multite_config['run_bayesian_fitting'] and bayesian_results:
        print(f"Bayesian fitting: {bayesian_results['successful_fits']}/{bayesian_results['total_processed']} voxels "
              f"({100 * bayesian_results['successful_fits'] / bayesian_results['total_processed']:.1f}%)")


    print("Multi-TE ASL processing completed!")

    # Store all results
    results.update({
        'tis': tis,
        'tes': tes,
        'ntes': ntes,
        'taus': taus,
        'config': multite_config
    })

    return results




if __name__ == "__main__":
    # Run full Multi-TE processing with both LS and Bayesian fitting
    results = asl_multite()

