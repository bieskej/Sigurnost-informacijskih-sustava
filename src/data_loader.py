import pandas as pd
import numpy as np
import os
import glob
from src.utils import setup_logger

logger = setup_logger('data_loader')

def load_data(data_path, sample_nrows=None):
    """
    Loads data from a directory of CSV files or a single CSV file.
    Args:
        data_path (str): Path to directory or file.
        sample_nrows (int, optional): Number of rows to read per file for testing.
    Returns:
        pd.DataFrame: Combined dataframe.
    """
    if os.path.isdir(data_path):
        all_files = glob.glob(os.path.join(data_path, "*.csv"))
        if not all_files:
            logger.warning(f"No CSV files found in {data_path}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(all_files)} CSV files in {data_path}")
        df_list = []
        for filename in all_files:
            logger.info(f"Loading {filename}...")
            try:
                # UNSW-NB15 often has just simple CSVs, sometimes with no header? 
                # Usually they have headers.
                df = pd.read_csv(filename, nrows=sample_nrows, skipinitialspace=True, encoding='latin1') # 'latin1' often needed for UNSW
                df_list.append(df)
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                
        if not df_list:
            return pd.DataFrame()
            
        combined_df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Combined data shape: {combined_df.shape}")
        return combined_df
        
    elif os.path.isfile(data_path):
        logger.info(f"Loading {data_path}...")
        try:
            df = pd.read_csv(data_path, nrows=sample_nrows, skipinitialspace=True, encoding='latin1')
            logger.info(f"Data shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading {data_path}: {e}")
            return pd.DataFrame()
    else:
        logger.error(f"Path not found: {data_path}")
        return pd.DataFrame()

def generate_synthetic_data(n_samples=1000, n_features=20, dataset_type='UNSW_NB15'):
    """
    Generates synthetic data mimicking UNSW-NB15.
    """
    logger.info(f"Generating synthetic data ({dataset_type}) with {n_samples} samples...")
    
    if dataset_type == 'UNSW_NB15':
        feature_names = [f"feat_{i}" for i in range(n_features)]
        # Add some UNSW specific columns
        feature_names.extend(['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl'])
        
        data = np.random.rand(n_samples, len(feature_names))
        labels = np.random.choice(['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconnaissance'], size=n_samples, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05])
        
        df = pd.DataFrame(data, columns=feature_names)
        df['attack_cat'] = labels
        df['label'] = (df['attack_cat'] != 'Normal').astype(int)
        
    else:
        # CICIDS style
        feature_names = [f"Feature_{i}" for i in range(n_features)]
        feature_names.extend(['Destination Port', 'Flow Duration', 'Total Fwd Packets'])
        data = np.random.rand(n_samples, len(feature_names))
        labels = np.random.choice(['BENIGN', 'DDoS', 'PortScan'], size=n_samples, p=[0.8, 0.1, 0.1])
        df = pd.DataFrame(data, columns=feature_names)
        df['Label'] = labels

    logger.info("Synthetic data generated.")
    return df
