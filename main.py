import argparse
import pandas as pd
import sys
from src.data_loader import load_data, generate_synthetic_data
from src.preprocessing import Preprocessor
from src.features import FeatureSelector
from src.models import get_model
from src.evaluation import Evaluator
from src.utils import setup_logger

logger = setup_logger('main')

def main():
    parser = argparse.ArgumentParser(description='Malicious Network Traffic Detection')
    parser.add_argument('--data_path', type=str, default='data/raw', help='Path to CSV data')
    parser.add_argument('--use_synthetic', action='store_true', help='Use generated synthetic data instead of loading from disk')
    parser.add_argument('--models', nargs='+', default=['rf', 'svm', 'mlp'], help='Models to train')
    parser.add_argument('--balance_method', default='smote', choices=['smote', 'under', 'none'], help='Method to handle class imbalance')
    parser.add_argument('--n_samples', type=int, default=10000, help='Max rows to read from CSV (for testing)')
    
    args = parser.parse_args()
    
    # 1. Load Data
    if args.use_synthetic:
        logger.info("Used --use_synthetic flag. Generating data...")
        df = generate_synthetic_data(n_samples=5000)
    else:
        df = load_data(args.data_path, sample_nrows=args.n_samples)
        if df.empty:
            logger.error("No data loaded. Please provide valid CSVs in data/raw or use --use_synthetic")
            sys.exit(1)

    # 2. Preprocessing
    processor = Preprocessor()
    df = processor.clean_data(df)
    
    try:
        y = processor.encode_target(df)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
        
    X_train, X_test, y_train, y_test = processor.split_and_scale(df, y)
    
    # 3. Feature Selection
    # (Optional: In a real run you might only run this once and save the feature list)
    selector = FeatureSelector()
    X_train = selector.remove_low_variance(X_train)
    X_train = selector.remove_highly_correlated(X_train)
    # Apply same feature drop to test set
    X_test = X_test[X_train.columns]
    
    # 4. Balancing (Training set only!)
    if args.balance_method != 'none':
        X_train_bal, y_train_bal = processor.balance_data(X_train, y_train, method=args.balance_method)
    else:
        X_train_bal, y_train_bal = X_train, y_train
        
    # 5. Model Training & Evaluation
    evaluator = Evaluator(output_dir='results')
    results = []
    
    class_names = list(processor.label_encoder.classes_)
    
    for model_name in args.models:
        logger.info(f"--- Training {model_name.upper()} ---")
        model = get_model(model_name)
        
        try:
            model.fit(X_train_bal, y_train_bal)
            res = evaluator.evaluate_model(model, X_test, y_test, model_name.upper(), class_names)
            results.append(res)
        except Exception as e:
            logger.error(f"Failed to train/evaluate {model_name}: {e}")
            
    # 6. Comparison
    if results:
        results_df = pd.DataFrame(results)
        print("\n=== Final Results ===")
        print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1']])
        
        evaluator.plot_comparison(results_df)

if __name__ == "__main__":
    main()
