"""
BASELINE MODELS FOR COMPARISON
4 baseline classifiers to compare against the cross-layer approach.
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from generate_dataset import FEATURE_NAMES

os.makedirs('results', exist_ok=True)

BASELINES = {
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=6, random_state=42),
    'SVM_RBF': SVC(kernel='rbf', C=10, gamma='scale',
                    class_weight='balanced', random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                          random_state=42),
    'LogisticRegression': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42),
}

FEATURE_SETS = {
    'Raw_Only_19': list(range(0, 19)),
    'With_CrossLayer_25': list(range(0, 25)),
    'Full_29': list(range(0, 29)),
}

def run_baselines():
    print("Loading data...")
    train_df = pd.read_csv('data/train_dataset.csv')
    test_df = pd.read_csv('data/test_dataset.csv')
    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df['label'].values
    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df['label'].values

    results = {}
    for model_name, model in BASELINES.items():
        for feat_name, feat_indices in FEATURE_SETS.items():
            key = f"{model_name}_{feat_name}"
            print(f"\nTraining {key}...")
            try:
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train[:, feat_indices], y_train)
                y_pred = model_copy.predict(X_test[:, feat_indices])
                f1_w = f1_score(y_test, y_pred, average='weighted')
                f1_c2 = f1_score(y_test, y_pred, average=None)
                print(f"  Weighted F1: {f1_w:.4f}")
                print(classification_report(y_test, y_pred,
                      target_names=['Normal', 'Single', 'Coordinated']))
                results[key] = {
                    'model': model_name, 'features': feat_name,
                    'n_features': len(feat_indices),
                    'weighted_f1': round(f1_w, 4),
                    'per_class_f1': [round(f, 4) for f in f1_c2.tolist()],
                }
            except Exception as e:
                print(f"  ERROR: {e}")
                results[key] = {'model': model_name, 'features': feat_name, 'error': str(e)}

    with open('results/baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nBaseline results saved to results/baseline_results.json")
    return results

if __name__ == '__main__':
    run_baselines()
