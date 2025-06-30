import nibabel as nib
import json
import os

def load_nifti_file(path):
    """Loads NIFTI file and returns the image and the data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = nib.load(path)
    data = img.get_fdata()
    return img, data

def load_json_metadata(json_path):
    """Loads metadata from JSON file"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    with open(json_path, "r") as f:
        metadata = json.load(f)
    return metadata
