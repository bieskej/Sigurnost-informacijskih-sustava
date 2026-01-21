from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from src.utils import setup_logger

logger = setup_logger('models')

def get_model(model_name, **kwargs):
    """
    Factory function to get models.
    """
    logger.info(f"Initializing {model_name}...")
    
    if model_name.lower() == 'rf':
        # Default parameters for Random Forest
        params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        params.update(kwargs)
        return RandomForestClassifier(**params)
    
    elif model_name.lower() == 'svm':
        # LinearSVC is faster for large datasets than SVC(kernel='linear')
        # For non-linear, we might need SVC or SGDClassifier with kernel approximation
        params = {'random_state': 42, 'dual': False} # dual=False for n_samples > n_features
        params.update(kwargs)
        return LinearSVC(**params)
    
    elif model_name.lower() == 'mlp':
        # Simple MLP
        params = {'hidden_layer_sizes': (64, 32), 'max_iter': 500, 'random_state': 42}
        params.update(kwargs)
        return MLPClassifier(**params)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")

if __name__ == "__main__":
    model = get_model('rf')
    print(model)
