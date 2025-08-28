# Estimation of Blood–Brain Barrier (BBB) Parameters from MRI

## Introduction
**Project**: Blood–Brain Barrier Module  
**Name**: Melissa Lange  
**Email**: langemelissa97@gmail.com  
**GitHub**: [https://github.com/limonenmelissa/osipi/tree/gsoc](https://github.com/limonenmelissa/osipi/tree/gsoc)  
**Organisation**: Open Science Initiative for Perfusion Imaging (OSIPI)  
**Mentors**: Ben Dickie, Jan Petr  
**Duration**: Google Summer of Code 2025  
---
## Project Overview
**Objective**: This project aims to estimate physiological parameters that characterise the permeability of the blood–brain barrier (BBB) using MRI data. As these parameters cannot be measured directly, they are inferred from the dynamics of Arterial Spin Labelling (ASL) signals.  

**Summary**: The project centres on modelling and solving the inverse problem of BBB permeability estimation. Alongside traditional least-squares fitting, a Bayesian inference framework has been implemented. This approach combines prior physiological knowledge with observed signals to derive the posterior distribution of the parameters.  

---

## Goals and Objectives
- Develop mathematical models describing the ASL signal: one- and two-compartment models for single-echo acquisitions, and a three-compartment model for multi-echo acquisitions.  
- Implement parameter estimation methods using least-squares fitting (`scipy.curve_fit`) and Bayesian inference (`stan`).  
- Provide an extensible framework that allows for the addition of further models or parameters in the future.  

---

## Work Completed

### Key Achievements
**Data Handling**: Functions for loading, reading, and saving NIfTI images.  
**Mathematical Modelling**: Implementation of one- and two-compartment models for single-echo data, extended to a three-compartment model for multi-echo data.  
**Estimation Methods**: Implementation of least-squares fitting with `scipy.curve_fit` and Bayesian inference using `stan`, incorporating physiological priors.  

### Technical Details
**Models**: One-, two-, and three-compartment models (based on [this paper](https://doi.org/10.1002/mrm.22320) for single-echo time and on [this paper](https://doi.org/10.3389/fnins.2021.719676), supporting both single- and multi-echo acquisitions.  
**Methods**: Least-squares fitting via `scipy`; Bayesian inference with `stan`, including posterior sampling and uncertainty quantification.  
**Code Contributions**: Core models and fitting methods are available on [GitHub](https://github.com/limonenmelissa/osipi/tree/gsoc).  
**Pending Work**: Extending models to incorporate additional physiological mechanisms and optimising the runtime of Bayesian inference.  

---

# Project Status

## Code Overview
- Python code:
  - **data_handling.py**  
    - functions for loading and processing NIfTI data
  - **deltaM_model.py**  
    - implementation of one- and two-compartment models
  - **multi_te_model.py**  
    - implementation of the three-compartment model
  - **fitting_single_te.py**  
    - data fitting for single-TE measurements
  - **fitting_multi_te.py**  
    - data fitting for multi-TE measurements
  - **single_te_asl.py**  
    - main script for single-TE analysis
  - **multi_te_asl.py**  
    - main script for multi-TE analysis
  - **config.json**
    - config file for all python files in this package 
  - **requirements.txt**
    - for easy installation of necessary Python packages 

## Usage examples
To run the code, make sure you have installed all necessary Python packages. They are given in `requirements.txt`.

For **single-echo time (single-TE) data**:  

1. Place your data in the following structure:  
<project_root>/data/1TE/
├── M0.nii
├── PWI4D.nii
├── M0.json
└── PWI4D.json  
2. Open `single_te_asl.py` to select the model:  
- Options: `simple` (one-compartment) or `extended` (two-compartment)  
- Default: `simple`
3. Run the script: python3 single_te_asl.py
4. Output: Fitted ATT and CBF NifTI images (and for the extended model additionally aBv and ATT_a) are saved in data/1TE. Both least-squares and Bayesian fitting are performed by default.


For **multi-echo time (single-TE) data**:  

1. Place your data in the following structure:  
<project_root>/data/multite/
├── M0.nii.gz
├── PWI4D.nii
├── M0.json
└── PWI4D.json  
2. Open `multi_te_asl.py`. For this script, only the three-compartment model is available. By default, both LS and Bayesian fitting will be done when running the script.
3. Run the script: python3 multi_te_asl.py
4. Output: Fitted ATT and CBF NifTI images are saved in /data/1TE.


## Completed Features
- Implemented one-, two-, and three-compartment models  
- Least-squares and Bayesian inference  
- Comparative analysis under noise conditions  

## In Progress
- Finalisation of documentation  
- Refinement and testing of the multi-echo time fitting algorithm  
- Finalisation of `debug_asl.py` for possible future use

---

## Future Work
Extending to alternative ASL models (e.g., diffusion-weighted models, see [Jin et al., 2020, DOI: 10.1002/mrm.27632](https://doi.org/10.1002/mrm.27632)) and improving efficiency for Bayesian fitting, particularly for multi-echo time models.  

---
## Lessons Learned
Enhanced Python programming fluency, gained hands-on experience with NIfTI data, and applied optimisation (`scipy`) as well as Bayesian inference (`stan`) techniques.  

---
## Acknowledgements
**Mentors**: A huge thanks to Ben Dickie and Jan Petr for their guidance and support throughout the project. I greatly appreciate your feedback and encouragement during this work. It was fun to work with you.  
**Organisation**: Many thanks to OSIPI and Google Summer of Code for providing this opportunity to contribute to medical imaging research.  

---
## Conclusion
**Summary**: This project successfully developed methods for estimating BBB parameters from ASL-MRI signals, combining compartment modelling with Bayesian inference. The framework enables robust quantification of physiological parameters from ASL-MRI data.
