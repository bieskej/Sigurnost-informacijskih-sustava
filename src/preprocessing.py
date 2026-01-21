import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from src.utils import setup_logger, DatasetConfig

logger = setup_logger('preprocessing')

class Preprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def clean_data(self, df):
        """
        Removes infinity/NaN values and duplicates.
        Also detects dataset type and standardizes column names if needed.
        """
        initial_shape = df.shape
        logger.info("Cleaning data...")
        
        # Detect dataset
        ds_type = DatasetConfig.detect_dataset(df)
        logger.info(f"Detected dataset type: {ds_type}")
        self.ds_config = getattr(DatasetConfig, ds_type, DatasetConfig.UNSW_NB15) # Default to UNSW if unknown
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # UNSW-NB15 specific cleanup
        if 'attack_cat' in df.columns:
            # Filled NaNs in attack_cat often imply 'Normal' in some CSV versions, 
            # but let's be careful. If label is 0, attack_cat should be Normal.
            df.loc[(df['label'] == 0) & (df['attack_cat'].isna()), 'attack_cat'] = 'Normal'
            # Any remaining NaNs -> drop
        
        # Replace Inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop NaN
        df = df.dropna()
        
        # Drop duplicates
        # df = df.drop_duplicates() # Sometimes risky with flow data if timestamps absent, but generally good
        
        logger.info(f"Data cleaned. Rows dropped: {initial_shape[0] - df.shape[0]}")
        return df
    
    def encode_target(self, df, use_binary=False):
        """
        Encodes the target string column into integers.
        """
        target_col = self.ds_config.LABEL_COLUMN
        
        if use_binary and hasattr(self.ds_config, 'BINARY_LABEL_COLUMN') and self.ds_config.BINARY_LABEL_COLUMN in df.columns:
            logger.info(f"Using binary label column '{self.ds_config.BINARY_LABEL_COLUMN}'")
            return df[self.ds_config.BINARY_LABEL_COLUMN].values
            
        if target_col not in df.columns:
            if 'Label' in df.columns: # Fallback
                target_col = 'Label'
            else:
                logger.error(f"Target column '{target_col}' not found.")
                raise ValueError(f"Target column '{target_col}' not found.")
            
        logger.info(f"Encoding target column '{target_col}'...")
        # Ensure strings
        df[target_col] = df[target_col].astype(str)
        y = self.label_encoder.fit_transform(df[target_col])
        logger.info(f"Classes found: {list(self.label_encoder.classes_)}")
        
        return y
    
    def split_and_scale(self, df, y):
        """
        Splits data into train/test and scales features.
        Also handles non-numeric columns like 'proto', 'service', 'state' in UNSW-NB15 by One-Hot Encoding.
        """
        target_col = self.ds_config.LABEL_COLUMN if self.ds_config.LABEL_COLUMN in df.columns else 'Label'
        if hasattr(self.ds_config, 'BINARY_LABEL_COLUMN') and self.ds_config.BINARY_LABEL_COLUMN in df.columns:
            drop_cols = [target_col, self.ds_config.BINARY_LABEL_COLUMN]
        else:
            drop_cols = [target_col]
            
        if 'id' in df.columns:
            drop_cols.append('id')
            
        X = df.drop(columns=drop_cols, errors='ignore')
        
        # Handle Categorical Features (e.g. proto, service, state)
        cat_cols = X.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            logger.info(f"One-hot encoding categorical features: {list(cat_cols)}")
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            
        # Ensure only numeric columns remain
        X = X.select_dtypes(include=[np.number])
        
        logger.info(f"Splitting data (test_size={self.test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info("Scaling features...")
        # Deal with potential column mismatch after splitting (rare but possible with get_dummies if categories missing in split)
        # Align columns to ensure both have same dummies
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def balance_data(self, X, y, method='smote'):
        """
        Balances the training dataset.
        """
        logger.info(f"Balancing data using {method}...")
        counter_before = pd.Series(y).value_counts()
        # logger.info(f"Class distribution before balancing:\n{counter_before}")
        
        # SMOTE can fail if k_neighbors > n_samples of minority class
        # Check smallest class text
        min_class_samples = counter_before.min()
        k_neighbors = 5
        if min_class_samples < 6:
            k_neighbors = max(1, min_class_samples - 1)
            logger.warning(f"Smallest class has {min_class_samples} samples. Adjusting SMOTE k_neighbors to {k_neighbors}")
        
        try:
            if method == 'smote':
                sampler = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
            elif method == 'under':
                sampler = RandomUnderSampler(random_state=self.random_state)
            else:
                return X, y
                
            X_res, y_res = sampler.fit_resample(X, y)
            return X_res, y_res
        except Exception as e:
            logger.error(f"Balancing failed: {e}. Proceeding without balancing.")
            return X, y

if __name__ == "__main__":
    from src.data_loader import generate_synthetic_data
    df = generate_synthetic_data(dataset_type='UNSW_NB15')
    proc = Preprocessor()
    df_clean = proc.clean_data(df)
    y = proc.encode_target(df_clean)
    X_train, X_test, y_train, y_test = proc.split_and_scale(df_clean, y)
    print("Train shape:", X_train.shape)
