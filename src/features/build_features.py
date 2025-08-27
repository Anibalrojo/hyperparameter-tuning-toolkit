import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add the project root to the Python path to allow importing from 'src'
# This goes up two levels from src/features to the project root
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

def process_pima_data(raw_data_path, processed_data_path):
    """
    Loads the raw Pima diabetes data, performs splitting and scaling,
    and saves the processed data to the specified directory.
    """
    print("Processing Pima Indians Diabetes dataset...")
    
    # Load raw data
    pima_df = pd.read_csv(os.path.join(raw_data_path, 'pima-indians-diabetes.csv'))
    
    # Separate features and target
    X = pima_df.drop('Outcome', axis=1)
    y = pima_df['Outcome']
    
    # Split the data (using the 80/20 split from the first notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames for saving
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Save the processed data
    os.makedirs(processed_data_path, exist_ok=True)
    X_train_scaled_df.to_csv(os.path.join(processed_data_path, 'pima_X_train.csv'), index=False)
    X_test_scaled_df.to_csv(os.path.join(processed_data_path, 'pima_X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_path, 'pima_y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_path, 'pima_y_test.csv'), index=False)
    
    print("Pima dataset processing complete.")

def process_breast_cancer_data(raw_data_path, processed_data_path):
    """
    Loads the raw Breast Cancer data, performs splitting and scaling,
    and saves the processed data to the specified directory.
    """
    print("Processing Breast Cancer Wisconsin dataset...")
    
    # Load raw data
    cancer_df = pd.read_csv(os.path.join(raw_data_path, 'breast_cancer.csv'))
    
    # Separate features and target
    X = cancer_df.drop('target', axis=1)
    y = cancer_df['target']
    
    # Split the data (using the 70/30 split from the third notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames for saving
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Save the processed data
    os.makedirs(processed_data_path, exist_ok=True)
    X_train_scaled_df.to_csv(os.path.join(processed_data_path, 'cancer_X_train.csv'), index=False)
    X_test_scaled_df.to_csv(os.path.join(processed_data_path, 'cancer_X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_path, 'cancer_y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_path, 'cancer_y_test.csv'), index=False)
    
    print("Breast Cancer dataset processing complete.")

def main():
    """
    Main function to run the data processing scripts.
    """
    # Define project directories
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    raw_data_path = os.path.join(project_dir, 'data', '01_raw')
    processed_data_path = os.path.join(project_dir, 'data', '02_processed')
    
    # Process the Pima dataset
    process_pima_data(raw_data_path, processed_data_path)
    
    # Process the Breast Cancer dataset
    process_breast_cancer_data(raw_data_path, processed_data_path)

if __name__ == '__main__':
    main()
