import logging
import os
import sys

def setup_logger(name, log_file='project.log', level=logging.INFO):
    """Function to setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(console_handler)
        
    return logger

class DatasetConfig:
    """
    Configuration for different datasets (CICIDS or UNSW-NB15).
    """
    class CICIDS:
        LABEL_COLUMN = 'Label'
        BENIGN_LABEL = 'BENIGN'
        # Feature names are usually PascalCase, e.g., 'Flow Duration'
        
    class UNSW_NB15:
        LABEL_COLUMN = 'attack_cat' # Or 'label' for binary, but let's use attack_cat for consistency with multi-class
        BINARY_LABEL_COLUMN = 'label'
        BENIGN_LABEL = 'Normal'
        # Some columns to drop if present (ids, etc.)
        DROP_COLS = ['id', 'attack_cat', 'label'] 

    @staticmethod
    def detect_dataset(df):
        """
        Heuristics to detect dataset type based on columns.
        """
        cols = set(df.columns)
        if 'attack_cat' in cols or 'label' in cols:
            return 'UNSW_NB15'
        elif ' Flow Duration' in cols or 'Flow Duration' in cols or ' Label' in cols or 'Label' in cols:
            return 'CICIDS'
        else:
            return 'UNKNOWN'

# Default to None, will be detected at runtime
CURRENT_CONFIG = DatasetConfig.UNSW_NB15 
