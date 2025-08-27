import os
import logging
from io import StringIO

import pandas as pd
import requests
from sklearn.datasets import load_breast_cancer


def download_pima_dataset(output_filepath):
    """Downloads the Pima Indians Diabetes dataset and saves it to a CSV file."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    
    # Column names from the original notebook
    column_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]

    logger = logging.getLogger(__name__)
    
    if os.path.exists(output_filepath):
        logger.info(f"'{os.path.basename(output_filepath)}' already exists. Skipping download.")
        return

    logger.info(f"Downloading Pima Indians Diabetes dataset from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Use StringIO to read the CSV content from the response text
        df = pd.read_csv(StringIO(response.text), header=None, names=column_names)
        df.to_csv(output_filepath, index=False)

        logger.info(f"Successfully saved dataset to '{output_filepath}'")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download dataset. Error: {e}")

def create_breast_cancer_dataset(output_filepath):
    """Loads the Breast Cancer Wisconsin dataset from scikit-learn and saves it to a CSV file."""
    logger = logging.getLogger(__name__)

    if os.path.exists(output_filepath):
        logger.info(f"'{os.path.basename(output_filepath)}' already exists. Skipping creation.")
        return

    logger.info("Loading Breast Cancer Wisconsin dataset from scikit-learn...")
    try:
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        
        df.to_csv(output_filepath, index=False)
        logger.info(f"Successfully saved dataset to '{output_filepath}'")

    except Exception as e:
        logger.error(f"Failed to create dataset. Error: {e}")


def main():
    """
    Runs data-fetching scripts to download and save the raw datasets
    into `data/01_raw`.
    """
    # Setup logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Define project directory and output paths
    # This script is in src/data, so we go up two levels to get to the project root
    project_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    raw_data_path = os.path.join(project_dir, 'data', '01_raw')

    # Ensure the output directory exists
    os.makedirs(raw_data_path, exist_ok=True)
    
    pima_output_path = os.path.join(raw_data_path, 'pima-indians-diabetes.csv')
    breast_cancer_output_path = os.path.join(raw_data_path, 'breast_cancer.csv')

    # Run the functions
    download_pima_dataset(pima_output_path)
    create_breast_cancer_dataset(breast_cancer_output_path)


if __name__ == '__main__':
    main()


