import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.utils import setup_logger

logger = setup_logger('evaluation')

class Evaluator:
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate_model(self, model, X_test, y_test, model_name="Model", class_names=None):
        """
        Evaluates a trained model and returns metrics.
        """
        logger.info(f"Evaluating {model_name}...")
        y_pred = model.predict(X_test)
        
        # Calculate individual metrics
        acc = accuracy_score(y_test, y_pred)
        # Macro average is useful for imbalanced/multi-class problems to treat classes equally
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        logger.info(f"{model_name} Results - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
        
        # Detailed report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, model_name, class_names)
        
        return {
            'Model': model_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'Report': report
        }

    def plot_confusion_matrix(self, cm, model_name, class_names):
        """
        Plots and saves the confusion matrix.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names if class_names is not None else "auto",
                    yticklabels=class_names if class_names is not None else "auto")
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/cm_{model_name}.png')
        plt.close()
        logger.info(f"Saved confusion matrix for {model_name}")

    def plot_comparison(self, results_df):
        """
        Plots a bar chart comparing models.
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        df_melted = results_df.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric')
        plt.title('Model Comparison')
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison.png')
        plt.close()
        logger.info("Saved model comparison plot")

if __name__ == "__main__":
    # Test evaluation
    from src.data_loader import generate_synthetic_data
    from src.preprocessing import Preprocessor
    from src.models import get_model
    
    # 1. Pipeline prep
    df = generate_synthetic_data(n_samples=200)
    proc = Preprocessor()
    df = proc.clean_data(df)
    y = proc.encode_target(df)
    X_train, X_test, y_train, y_test = proc.split_and_scale(df, y)
    
    # 2. Train dummy model
    model = get_model('rf', n_estimators=10)
    model.fit(X_train, y_train)
    
    # 3. Evaluate
    evaluator = Evaluator()
    evaluator.evaluate_model(model, X_test, y_test, "RandomForest_Test", class_names=list(proc.label_encoder.classes_))
