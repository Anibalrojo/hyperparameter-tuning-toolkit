import os
import pytest
import sys

# Add the 'src' directory to the Python path to allow importing 'make_dataset'
# This goes up one level from 'tests' to the project root
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.data import make_dataset

def test_make_dataset_creates_files():
    """
    Tests the main data creation script to ensure it creates the expected output files.
    """
    # Define the expected output file paths
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pima_path = os.path.join(project_dir, 'data', '01_raw', 'pima-indians-diabetes.csv')
    cancer_path = os.path.join(project_dir, 'data', '01_raw', 'breast_cancer.csv')

    # Ensure files don't exist before running the script (for a clean test)
    if os.path.exists(pima_path):
        os.remove(pima_path)
    if os.path.exists(cancer_path):
        os.remove(cancer_path)
    
    # Assert that the files are gone before we start
    assert not os.path.exists(pima_path), "Pre-test cleanup failed for Pima dataset."
    assert not os.path.exists(cancer_path), "Pre-test cleanup failed for Breast Cancer dataset."

    # Run the main data creation script
    make_dataset.main()

    # Assert that the files have been created
    assert os.path.exists(pima_path), "make_dataset.py did not create the Pima dataset."
    assert os.path.exists(cancer_path), "make_dataset.py did not create the Breast Cancer dataset."

    # Optional: Clean up the created files after the test
    os.remove(pima_path)
    os.remove(cancer_path)


