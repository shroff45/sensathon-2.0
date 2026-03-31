"""
TRAINING PIPELINE
- 5-fold Stratified Group Cross-Validation
- Ablation study with confidence intervals
- Statistical significance testing
- Per-attack-type analysis
- Feature importance analysis
- Model export for ESP32
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (f1_score, classification_report, confusion_matrix, precision_recall_fscore_support)
from scipy import stats
from generate_dataset import FEATURE_NAMES

MODEL_PARAMS = {
    'n_estimators': 15, 'max_depth': 10, 'min_samples_leaf': 8,
    'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1,
}

FEATURE_CONFIGS = {
    'Sensor_Only': list(range(0, 7)),
    'CAN_Only': list(range(7, 14)),
    'V2X_Only': list(range(14, 19)),
    'CrossLayer_Only': list(range(19, 25)),
    'All_Raw_No_Cross': list(range(0, 19)),
    'Full_25': list(range(0, 25)),
    'Full_29': list(range(0, 29)),
}

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def load_data():
    print("Loading datasets...")
    train_df = pd.read_csv('data/train_dataset.csv')
    test_df = pd.read_csv('data/test_dataset.csv')
    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df['label'].values
    groups_train = train_df['scenario_id'].values
    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df['label'].values
    print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test:     {X_test.shape[0]} samples")
    print(f"Train class dist: {np.bincount(y_train)}")
    print(f"Test class dist:  {np.bincount(y_test)}")
    return X_train, y_train, groups_train, X_test, y_test, train_df, test_df

def run_ablation_cv(X, y, groups, n_splits=5):
    print("\n" + "=" * 70)
    print("ABLATION STUDY WITH 5-FOLD STRATIFIED GROUP CROSS-VALIDATION")
    print("=" * 70)
    results = {}
    for config_name, feature_indices in FEATURE_CONFIGS.items():
        fold_scores_weighted = []
        fold_scores_class2 = []
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            X_train_fold = X[train_idx][:, feature_indices]
            X_val_fold = X[val_idx][:, feature_indices]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            model = RandomForestClassifier(**MODEL_PARAMS)
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            f1_w = f1_score(y_val_fold, y_pred, average='weighted')
            p, r, f, s = precision_recall_fscore_support(y_val_fold, y_pred, labels=[2], zero_division=0)
            f1_c2 = f[0] if len(f) > 0 else 0
            fold_scores_weighted.append(f1_w)
            fold_scores_class2.append(f1_c2)
        mean_w = np.mean(fold_scores_weighted)
        std_w = np.std(fold_scores_weighted)
        ci_w = stats.t.interval(0.95, n_splits - 1, loc=mean_w, scale=stats.sem(fold_scores_weighted))
        mean_c2 = np.mean(fold_scores_class2)
        std_c2 = np.std(fold_scores_class2)
        ci_c2 = stats.t.interval(0.95, n_splits - 1, loc=mean_c2, scale=stats.sem(fold_scores_class2))
        results[config_name] = {
            'features': feature_indices, 'n_features': len(feature_indices),
            'weighted_f1_mean': round(mean_w, 4), 'weighted_f1_std': round(std_w, 4),
            'weighted_f1_ci': (round(ci_w[0], 4), round(ci_w[1], 4)),
            'coord_f1_mean': round(mean_c2, 4), 'coord_f1_std': round(std_c2, 4),
            'coord_f1_ci': (round(ci_c2[0], 4), round(ci_c2[1], 4)),
            'fold_scores_weighted': fold_scores_weighted, 'fold_scores_class2': fold_scores_class2,
        }
        print(f"\n{config_name} ({len(feature_indices)} features):")
        print(f"  Weighted F1: {mean_w:.4f} +/- {std_w:.4f}  CI: [{ci_w[0]:.4f}, {ci_w[1]:.4f}]")
        print(f"  Coord F1:    {mean_c2:.4f} +/- {std_c2:.4f}  CI: [{ci_c2[0]:.4f}, {ci_c2[1]:.4f}]")
    print("\n" + "-" * 60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("-" * 60)
    for comparison in [('Full_29', 'All_Raw_No_Cross'), ('Full_29', 'Full_25'), ('Full_25', 'All_Raw_No_Cross')]:
        a_name, b_name = comparison
        if a_name in results and b_name in results:
            a_scores = results[a_name]['fold_scores_weighted']
            b_scores = results[b_name]['fold_scores_weighted']
            t_stat, p_value = stats.ttest_rel(a_scores, b_scores)
            sig = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
            print(f"  {a_name} vs {b_name}: t={t_stat:.4f}, p={p_value:.4f} -> {sig}")
    return results

def train_final_models(X_train, y_train, X_test, y_test, test_df):
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODELS ON FULL TRAINING DATA")
    print("=" * 70)
    models = {}
    for config_name, feature_indices in FEATURE_CONFIGS.items():
        print(f"\nTraining {config_name}...")
        model = RandomForestClassifier(**MODEL_PARAMS)
        model.fit(X_train[:, feature_indices], y_train)
        y_pred = model.predict(X_test[:, feature_indices])
        f1_w = f1_score(y_test, y_pred, average='weighted')
        print(f"  Test Weighted F1: {f1_w:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Single Attack', 'Coordinated Attack']))
        model_path = f'models/{config_name.lower()}_model.pkl'
        joblib.dump(model, model_path)
        models[config_name] = model
        if config_name == 'Full_29':
            print("\n" + "-" * 70)
            print("PER-ATTACK-TYPE DETECTION PERFORMANCE (Full_29 Model)")
            print("-" * 70)
            attack_results = {}
            for attack_name in sorted(test_df['attack_name'].unique()):
                mask = test_df['attack_name'].values == attack_name
                y_true_sub = y_test[mask]
                y_pred_sub = y_pred[mask]
                total = len(y_true_sub)
                if total == 0: continue
                if attack_name == 'normal':
                    fp = (y_pred_sub > 0).sum()
                    fp_rate = fp / total * 100
                    print(f"  {attack_name:35s} | FP Rate: {fp_rate:5.1f}% | n={total}")
                    attack_results[attack_name] = {'fp_rate': fp_rate, 'n': total}
                else:
                    detected = (y_pred_sub > 0).sum()
                    det_rate = detected / total * 100
                    correct = (y_true_sub == y_pred_sub).sum()
                    acc = correct / total * 100
                    unseen = "(UNSEEN)" if 'can_imu' in attack_name or 'speed_all' in attack_name else ""
                    print(f"  {attack_name:35s} | Det: {det_rate:5.1f}% | Acc: {acc:5.1f}% | n={total} {unseen}")
                    attack_results[attack_name] = {'detection_rate': det_rate, 'accuracy': acc, 'n': total}
            with open('results/per_attack_results.json', 'w') as f:
                json.dump(attack_results, f, indent=2)
    return models

def analyze_feature_importance(model, feature_names):
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    importance_data = []
    for rank, idx in enumerate(indices):
        name = feature_names[idx]
        imp = importances[idx]
        if idx < 7: layer = 'Sensor'
        elif idx < 14: layer = 'CAN'
        elif idx < 19: layer = 'V2X'
        elif idx < 25: layer = 'CrossLayer'
        else: layer = 'Temporal'
        importance_data.append({'rank': rank + 1, 'feature_index': int(idx), 'feature_name': name, 'importance': round(float(imp), 6), 'layer': layer})
        print(f"  {rank+1:2d}. [{layer:10s}] {name:30s} = {imp:.6f}")
    layer_imp = {}
    for d in importance_data:
        layer = d['layer']
        layer_imp[layer] = layer_imp.get(layer, 0) + d['importance']
    print(f"\nLayer-level importance:")
    for layer, imp in sorted(layer_imp.items(), key=lambda x: -x[1]):
        print(f"  {layer:12s}: {imp:.4f} ({imp*100:.1f}%)")
    with open('results/feature_importance.json', 'w') as f:
        json.dump(importance_data, f, indent=2)
    return importance_data

def false_positive_analysis(model, feature_indices):
    print("\n" + "=" * 70)
    print("FALSE POSITIVE STRESS TEST")
    print("=" * 70)
    from generate_dataset import scenario_emergency_brake, scenario_sharp_turn, generate_scenario_data
    stress_scenarios = {'Emergency Brake': scenario_emergency_brake, 'Sharp Turn': scenario_sharp_turn}
    for name, scenario_fn in stress_scenarios.items():
        all_preds = []
        for i in range(50):
            np.random.seed(1000 + i)
            rows = generate_scenario_data(scenario_fn, attack_fn=None, attack_label=0, scenario_id=9000+i)
            df = pd.DataFrame(rows)
            X = df[FEATURE_NAMES].values[:, feature_indices]
            preds = model.predict(X)
            all_preds.extend(preds)
        all_preds = np.array(all_preds)
        fp_count = (all_preds > 0).sum()
        fp_rate = fp_count / len(all_preds) * 100
        print(f"  {name:20s} | FP Rate: {fp_rate:.2f}% ({fp_count}/{len(all_preds)})")

if __name__ == '__main__':
    X_train, y_train, groups_train, X_test, y_test, train_df, test_df = load_data()
    ablation_results = run_ablation_cv(X_train, y_train, groups_train)
    ablation_save = {}
    for k, v in ablation_results.items():
        ablation_save[k] = {
            'n_features': v['n_features'], 'weighted_f1_mean': v['weighted_f1_mean'],
            'weighted_f1_std': v['weighted_f1_std'], 'weighted_f1_ci': v['weighted_f1_ci'],
            'coord_f1_mean': v['coord_f1_mean'], 'coord_f1_std': v['coord_f1_std'],
            'coord_f1_ci': v['coord_f1_ci'],
        }
    with open('results/ablation_results.json', 'w') as f:
        json.dump(ablation_save, f, indent=2)
    models = train_final_models(X_train, y_train, X_test, y_test, test_df)
    full_model = models['Full_29']
    importance_data = analyze_feature_importance(full_model, FEATURE_NAMES)
    false_positive_analysis(full_model, FEATURE_CONFIGS['Full_29'])
    y_pred_full = full_model.predict(X_test[:, FEATURE_CONFIGS['Full_29']])
    cm = confusion_matrix(y_test, y_pred_full)
    np.save('results/confusion_matrix.npy', cm)
    print("\n" + "=" * 70)
    print("FINAL CONFUSION MATRIX")
    print("=" * 70)
    print(f"              Pred_Normal  Pred_Single  Pred_Coord")
    print(f"True_Normal:    {cm[0][0]:7d}      {cm[0][1]:7d}      {cm[0][2]:7d}")
    print(f"True_Single:    {cm[1][0]:7d}      {cm[1][1]:7d}      {cm[1][2]:7d}")
    print(f"True_Coord:     {cm[2][0]:7d}      {cm[2][1]:7d}      {cm[2][2]:7d}")
    print("\n Training complete! Models saved to models/")
    print("   Results saved to results/")

    # M4: Run baselines for comparison data (dashboard Ablation tab)
    try:
        from baselines import run_baselines
        print("\n" + "=" * 70)
        print("RUNNING BASELINE COMPARISONS")
        print("=" * 70)
        run_baselines()
    except Exception as e:
        print(f"  Baseline comparison skipped: {e}")
