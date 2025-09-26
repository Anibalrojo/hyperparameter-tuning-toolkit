# Data Directory

This directory contains all datasets used in the hyperparameter tuning experiments.

## Directory Structure

- `01_raw/`: Original, immutable data
  - `breast_cancer.csv`: Breast Cancer Wisconsin dataset
  - `pima-indians-diabetes.csv`: Pima Indians Diabetes dataset

- `02_processed/`: Cleaned and processed data ready for modeling
  - `cancer_X_train.csv`, `cancer_X_test.csv`, `cancer_y_train.csv`, `cancer_y_test.csv`: Train/test splits for the Breast Cancer dataset
  - `pima_X_train.csv`, `pima_X_test.csv`, `pima_y_train.csv`, `pima_y_test.csv`: Train/test splits for the Pima Indians Diabetes dataset

## Data Generation

The raw data files are not included in the repository due to size constraints. To download and prepare the datasets, run:

```bash
python src/data/make_dataset.py
```

This script will:
1. Download the Pima Indians Diabetes dataset from an external source
2. Load the Breast Cancer Wisconsin dataset from scikit-learn
3. Save both datasets to the `01_raw` directory
4. Process the data and save train/test splits to the `02_processed` directory

Note: The processed data files are generated during the execution of the notebooks and are also excluded from version control.

