import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from src.utils import setup_logger

logger = setup_logger('features')

class FeatureSelector:
    def __init__(self, variance_threshold=0.01, correlation_threshold=0.95):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        self.selector = None

    def remove_low_variance(self, X):
        """
        Removes features with variance lower than threshold.
        """
        logger.info(f"Removing features with variance < {self.variance_threshold}...")
        selector = VarianceThreshold(threshold=self.variance_threshold)
        
        # Fit on dataframe
        selector.fit(X)
        
        # Get columns to keep
        concol = [column for column in X.columns 
                  if column not in X.columns[selector.get_support()]]
        
        X_reduced = X.drop(columns=concol)
        logger.info(f"Removed {len(concol)} low-variance features.")
        return X_reduced

    def remove_highly_correlated(self, X):
        """
        Removes highly correlated features.
        """
        logger.info(f"Removing features with correlation > {self.correlation_threshold}...")
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        
        X_reduced = X.drop(columns=to_drop)
        logger.info(f"Removed {len(to_drop)} highly correlated features.")
        return X_reduced

    def select_important_features(self, X, y, k=20):
        """
        Selects top k features based on Random Forest importance.
        """
        logger.info(f"Selecting top {k} features using Random Forest...")
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        top_k = importances.nlargest(k)
        
        logger.info(f"Top 5 features: {top_k.head().index.tolist()}")
        
        self.selected_features = top_k.index.tolist()
        return X[self.selected_features]

if __name__ == "__main__":
    from src.data_loader import generate_synthetic_data
    from src.preprocessing import Preprocessor
    
    df = generate_synthetic_data(n_features=50) # more features to test selection
    proc = Preprocessor()
    df = proc.clean_data(df)
    y = proc.encode_target(df)
    X, _, y_train, _ = proc.split_and_scale(df, y)
    
    selector = FeatureSelector()
    X = selector.remove_low_variance(X)
    X = selector.remove_highly_correlated(X)
    X_selected = selector.select_important_features(X, y_train, k=10)
    
    print("Selected features shape:", X_selected.shape)
