# Malicious Network Traffic Detection - Walkthrough

This document outlines how to use the implemented Intrusion Detection System (IDS).

## 1. Project Setup Setup
Ensure you are in the project root directory and have dependencies installed:
```bash
pip install -r requirements.txt
```

## 2. Running with Synthetic Data
To verify the system works without downloading large datasets, use the `--use_synthetic` flag. This generates random network flow data.

```bash
python main.py --use_synthetic --models rf svm mlp
```

**Expected Output:**
- Logs showing data generation, cleaning, and balancing.
- Training progress for Random Forest (RF), SVM, and MLP.
- Final results table comparing Accuracy, Precision, Recall, and F1-score.
- **Artifacts**: Check the `results/` folder for:
    - `cm_RF.png`, `cm_SVM.png`, etc. (Confusion Matrices)
    - `model_comparison.png` (Bar chart comparison)

## 3. Running with Real Data (CICIDS2017/UNSW-NB15)
1. Download the CSV files from the official source (e.g., CIC datasets).
2. Place the CSV files in `data/raw/`.
3. Run the pipeline:

```bash
python main.py --data_path data/raw --models rf
```

## 4. Key Components
- **`src/preprocessing.py`**: Handles cleaning, label encoding, and SMOTE balancing.
- **`src/features.py`**: Removes low variance/correlated features and selects top features using Random Forest importance.
- **`src/models.py`**: Definitions for RF, SVM, and MLP classifiers.
- **`src/evaluation.py`**: Generates classification reports and visualization plots.

## 5. Interpreting Results
- **Precision**: How many detected attacks were actually attacks?
- **Recall**: How many actual attacks did we catch? (Critical for security!)
- **F1-Score**: Balance between Precision and Recall.
- **Confusion Matrix**: Visualizes misclassifications (e.g., distinguishing between DDoS and PortScan).
