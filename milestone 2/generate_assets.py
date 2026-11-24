"""
generate_assets.py
==================
NETW504 Random Signals and Noise - Milestone 2
Team Name: Hloz
Author: Mazen Mohamed Hamdy Altelbany (ID: 64-12371) & Team Hloz

This script performs all data preparation, Bayesian estimation, model training,
validation selection, test evaluation, and figure generation for Milestone 2.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Set visual style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
DATA_PATH = os.path.join(BASE_DIR, "..", "data set", "Churn_Modelling.csv")
POSTERIORS_CSV = os.path.join(BASE_DIR, "Hloz_M2_Bayesian_Posteriors.csv")
os.makedirs(FIGURES_DIR, exist_ok=True)

print("--- Step 1: Loading Dataset & Splitting ---")
df_raw = pd.read_csv(DATA_PATH)
print(f"Loaded raw dataset with shape: {df_raw.shape}")

# Drop ID columns
drop_cols = ['RowNumber', 'CustomerId', 'Surname']
df = df_raw.drop(columns=drop_cols)
print(f"Dropped ID columns {drop_cols}. Remaining shape: {df.shape}")

# Define feature columns and target
target_col = 'Exited'
feature_cols = [c for c in df.columns if c != target_col]
numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
categorical_cols = ['Geography', 'Gender']

X = df[feature_cols].copy()
y = df[target_col].copy()

# Stratified split: 70% Train, 15% Validation, 15% Test
# First split: 70% train vs 30% temp
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
# Second split: split 30% temp into 50% val (15% total) and 50% test (15% total)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train split: {X_train_raw.shape[0]} samples (Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()})")
print(f"Val split:   {X_val_raw.shape[0]} samples (Class 0: {(y_val == 0).sum()}, Class 1: {(y_val == 1).sum()})")
print(f"Test split:  {X_test_raw.shape[0]} samples (Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()})")

print("\n--- Step 2: Preprocessing (Fit on Train ONLY) ---")
# Preprocessor pipeline: Impute + Scale numeric, Impute + One-hot categorical
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_cols),
    ('cat', cat_transformer, categorical_cols)
])

# Fit on training data ONLY
preprocessor.fit(X_train_raw)

# Get transformed feature names
ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(categorical_cols).tolist()
all_feature_names = numeric_cols + ohe_feature_names
print(f"Processed feature names ({len(all_feature_names)}): {all_feature_names}")

# Transform splits
X_train = preprocessor.transform(X_train_raw)
X_val = preprocessor.transform(X_val_raw)
X_test = preprocessor.transform(X_test_raw)

# Convert to DataFrames for easier indexing
df_X_train = pd.DataFrame(X_train, columns=all_feature_names, index=X_train_raw.index)
df_X_val = pd.DataFrame(X_val, columns=all_feature_names, index=X_val_raw.index)
df_X_test = pd.DataFrame(X_test, columns=all_feature_names, index=X_test_raw.index)

print("\n--- Step 3: Bayesian Estimation from Scratch ---")
# Empirical Class Priors from Training Set
N_train = len(y_train)
N_0 = (y_train == 0).sum()
N_1 = (y_train == 1).sum()
prior_0 = N_0 / N_train
prior_1 = N_1 / N_train
priors = {0: prior_0, 1: prior_1}
print(f"Empirical Priors: P(Y=0) = {prior_0:.4f} ({N_0}/{N_train}), P(Y=1) = {prior_1:.4f} ({N_1}/{N_train})")

# Feature Likelihood Parameters: Means and Variances per Class
# Gaussian Naive Bayes likelihood: p(x_j | Y=k) = 1 / sqrt(2*pi*sigma_kj^2) * exp(-(x_j - mu_kj)^2 / (2*sigma_kj^2))
means = {}
variances = {}
for k in [0, 1]:
    X_k = X_train[y_train.values == k]
    means[k] = np.mean(X_k, axis=0)
    # Variance smoothing epsilon to prevent division by zero / underflow, exactly as in sklearn GaussianNB
    var_k = np.var(X_k, axis=0)
    var_smoothing = 1e-9 * np.max(np.var(X_train, axis=0))
    variances[k] = var_k + var_smoothing

def calculate_gaussian_log_likelihood(x, mu, var):
    """
    Computes log p(x | mu, var) for independent features:
    log p(x | mu, var) = -0.5 * log(2 * pi * var) - (x - mu)^2 / (2 * var)
    """
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mu) ** 2) / var

def predict_bayes_scratch(X_mat):
    """
    Computes exact posterior probabilities P(Y=k | x) using Bayes' theorem:
    score_k(x) = P(Y=k) * prod_j p(x_j | Y=k)
    Computed via log-sum-exp for numerical stability.
    """
    N_samples = X_mat.shape[0]
    log_scores = np.zeros((N_samples, 2))
    
    for k in [0, 1]:
        log_prior = np.log(priors[k])
        # Sum of log likelihoods over all features
        log_liks = calculate_gaussian_log_likelihood(X_mat, means[k], variances[k])
        log_scores[:, k] = log_prior + np.sum(log_liks, axis=1)
        
    # Stable Softmax (Log-Sum-Exp trick)
    max_log = np.max(log_scores, axis=1, keepdims=True)
    exp_scores = np.exp(log_scores - max_log)
    posteriors = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    preds = np.argmax(posteriors, axis=1)
    return posteriors, preds

# Compute posteriors for all splits
post_train, preds_train = predict_bayes_scratch(X_train)
post_val, preds_val = predict_bayes_scratch(X_val)
post_test, preds_test = predict_bayes_scratch(X_test)

# Verify probabilities sum to 1.0
assert np.allclose(np.sum(post_train, axis=1), 1.0), "Train posteriors do not sum to 1!"
assert np.allclose(np.sum(post_test, axis=1), 1.0), "Test posteriors do not sum to 1!"
print("Posteriors calculated successfully! All row probabilities strictly sum to 1.0.")

# Construct Full Bayesian Posteriors Table
records = []
for idx, split_name, df_raw_subset, y_subset, post_subset, pred_subset in [
    (X_train_raw.index, 'Train', df_raw.loc[X_train_raw.index], y_train, post_train, preds_train),
    (X_val_raw.index, 'Validation', df_raw.loc[X_val_raw.index], y_val, post_val, preds_val),
    (X_test_raw.index, 'Test', df_raw.loc[X_test_raw.index], y_test, post_test, preds_test)
]:
    for i in range(len(idx)):
        records.append({
            'Sample_ID': idx[i],
            'Split': split_name,
            'Actual_Class': y_subset.iloc[i],
            'P(Y=0|X)': post_subset[i, 0],
            'P(Y=1|X)': post_subset[i, 1],
            'Predicted_Class': pred_subset[i]
        })

df_posteriors = pd.DataFrame(records)
df_posteriors.to_csv(POSTERIORS_CSV, index=False)
print(f"Saved full posteriors table to {POSTERIORS_CSV} ({len(df_posteriors)} rows).")

print("\n--- Step 4: 20 Test Samples & Worked Bayes Calculation ---")
# Select 20 diverse test samples: 10 actual class 0, 10 actual class 1
# Include clear cases, borderline cases, and our specific worked example (Sample 5750)
test_indices_0 = np.where(y_test.values == 0)[0][:10]

# For class 1, find sample 5750 position if in test set, or pick 10 diverse class 1 samples including 5750
test_idx_pos_5750 = np.where(X_test_raw.index == 5750)[0]
if len(test_idx_pos_5750) > 0:
    pos_5750 = test_idx_pos_5750[0]
    other_c1 = [idx for idx in np.where(y_test.values == 1)[0] if idx != pos_5750][:9]
    test_indices_1 = np.array([pos_5750] + other_c1)
else:
    test_indices_1 = np.where(y_test.values == 1)[0][:10]

selected_20_indices = np.concatenate([test_indices_0, test_indices_1])

# Build the 20-sample detailed table
sample_20_records = []
for idx_pos in selected_20_indices:
    orig_id = int(X_test_raw.index[idx_pos])
    raw_row = df_raw.loc[orig_id]
    scaled_feats = X_test[idx_pos]
    p0 = float(post_test[idx_pos, 0])
    p1 = float(post_test[idx_pos, 1])
    actual = int(y_test.iloc[idx_pos])
    pred = int(preds_test[idx_pos])
    
    rec = {
        'Sample_ID': orig_id,
        'Actual_Class': actual,
        'Predicted_Class': pred,
        'Prior_P(Y=0)': round(float(prior_0), 4),
        'Prior_P(Y=1)': round(float(prior_1), 4),
        'Posterior_P(Y=0|X)': round(p0, 4),
        'Posterior_P(Y=1|X)': round(p1, 4),
        'CreditScore': int(raw_row['CreditScore']),
        'Geography': str(raw_row['Geography']),
        'Gender': str(raw_row['Gender']),
        'Age': int(raw_row['Age']),
        'Tenure': int(raw_row['Tenure']),
        'Balance': round(float(raw_row['Balance']), 2),
        'NumOfProducts': int(raw_row['NumOfProducts']),
        'HasCrCard': int(raw_row['HasCrCard']),
        'IsActiveMember': int(raw_row['IsActiveMember']),
        'EstimatedSalary': round(float(raw_row['EstimatedSalary']), 2)
    }
    sample_20_records.append(rec)

df_20_samples = pd.DataFrame(sample_20_records)
print("Selected 20 test samples (10 class 0, 10 class 1):")
print(df_20_samples[['Sample_ID', 'Actual_Class', 'Predicted_Class', 'Posterior_P(Y=0|X)', 'Posterior_P(Y=1|X)']].head())

# Select worked example: Sample 5750 (the first in the class 1 list)
worked_idx_pos = test_indices_1[0]
worked_orig_id = int(X_test_raw.index[worked_idx_pos])
worked_raw = df_raw.loc[worked_orig_id]
worked_scaled = X_test[worked_idx_pos]

worked_details = []
for j, feat_name in enumerate(all_feature_names):
    x_val = float(worked_scaled[j])
    m0, v0 = float(means[0][j]), float(variances[0][j])
    m1, v1 = float(means[1][j]), float(variances[1][j])
    
    lik0 = float((1.0 / np.sqrt(2 * np.pi * v0)) * np.exp(-0.5 * ((x_val - m0) ** 2) / v0))
    lik1 = float((1.0 / np.sqrt(2 * np.pi * v1)) * np.exp(-0.5 * ((x_val - m1) ** 2) / v1))
    
    log_lik0 = float(-0.5 * np.log(2 * np.pi * v0) - 0.5 * ((x_val - m0) ** 2) / v0)
    log_lik1 = float(-0.5 * np.log(2 * np.pi * v1) - 0.5 * ((x_val - m1) ** 2) / v1)
    
    lik_ratio = float(lik1 / (lik0 + 1e-12))
    
    worked_details.append({
        'Feature': feat_name,
        'Scaled_Value': round(x_val, 4),
        'Mean_Y0': round(m0, 4),
        'Var_Y0': round(v0, 4),
        'Likelihood_Y0': float(lik0),
        'Log_Lik_Y0': round(log_lik0, 4),
        'Mean_Y1': round(m1, 4),
        'Var_Y1': round(v1, 4),
        'Likelihood_Y1': float(lik1),
        'Log_Lik_Y1': round(log_lik1, 4),
        'Likelihood_Ratio_Y1_vs_Y0': round(lik_ratio, 4)
    })

df_worked = pd.DataFrame(worked_details)
print(f"\nWorked Example for Sample ID {worked_orig_id} (Actual Class: {worked_raw['Exited']}):")
print(f"Priors: P(Y=0)={prior_0:.4f}, P(Y=1)={prior_1:.4f}")
print(f"Direct Posteriors: P(Y=0|X)={post_test[worked_idx_pos, 0]:.4f}, P(Y=1|X)={post_test[worked_idx_pos, 1]:.4f}")

print("\n--- Step 5: Training Conventional & Bayesian Models ---")
# 1. Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_test_probs = gnb.predict_proba(X_test)
gnb_test_preds = gnb.predict(X_test)

# Verify scratch Bayes vs sklearn GaussianNB
max_diff = np.max(np.abs(post_test - gnb_test_probs))
print(f"Max absolute difference between Scratch Bayes & sklearn GaussianNB: {max_diff:.2e}")

# 2. Decision Tree Hyperparameter Tuning on Validation Set
tree_depths = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
val_f1_trees = []
train_f1_trees = []
for d in tree_depths:
    dt_temp = DecisionTreeClassifier(max_depth=d, min_samples_split=20, min_samples_leaf=10, random_state=42)
    dt_temp.fit(X_train, y_train)
    val_preds = dt_temp.predict(X_val)
    train_preds = dt_temp.predict(X_train)
    val_f1_trees.append(f1_score(y_val, val_preds, average='macro'))
    train_f1_trees.append(f1_score(y_train, train_preds, average='macro'))

best_depth = tree_depths[np.argmax(val_f1_trees)]
print(f"Best Decision Tree max_depth on Validation Set: {best_depth} (Macro F1 = {max(val_f1_trees):.4f})")

best_dt = DecisionTreeClassifier(max_depth=best_depth, min_samples_split=20, min_samples_leaf=10, random_state=42)
best_dt.fit(X_train, y_train)
dt_test_probs = best_dt.predict_proba(X_test)
dt_test_preds = best_dt.predict(X_test)

# 3. Logistic Regression Hyperparameter Tuning on Validation Set
C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
val_f1_lr = []
train_f1_lr = []
for c_val in C_values:
    lr_temp = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
    lr_temp.fit(X_train, y_train)
    val_preds = lr_temp.predict(X_val)
    train_preds = lr_temp.predict(X_train)
    val_f1_lr.append(f1_score(y_val, val_preds, average='macro'))
    train_f1_lr.append(f1_score(y_train, train_preds, average='macro'))

best_C = C_values[np.argmax(val_f1_lr)]
print(f"Best Logistic Regression C on Validation Set: {best_C} (Macro F1 = {max(val_f1_lr):.4f})")

best_lr = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
best_lr.fit(X_train, y_train)
lr_test_probs = best_lr.predict_proba(X_test)
lr_test_preds = best_lr.predict(X_test)

print("\n--- Step 6: Evaluation on Untouched Test Set ---")
models = {
    'Direct Bayes (Scratch)': preds_test,
    'Gaussian Naive Bayes': gnb_test_preds,
    f'Decision Tree (depth={best_depth})': dt_test_preds,
    f'Logistic Regression (C={best_C})': lr_test_preds
}

eval_rows = []
for name, preds in models.items():
    acc = float(accuracy_score(y_test, preds))
    prec_macro = float(precision_score(y_test, preds, average='macro', zero_division=0))
    rec_macro = float(recall_score(y_test, preds, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_test, preds, average='macro', zero_division=0))
    f1_c0 = float(f1_score(y_test, preds, pos_label=0))
    f1_c1 = float(f1_score(y_test, preds, pos_label=1))
    
    eval_rows.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Macro Precision': round(prec_macro, 4),
        'Macro Recall': round(rec_macro, 4),
        'Macro F1': round(f1_macro, 4),
        'Class 0 F1 (Stayed)': round(f1_c0, 4),
        'Class 1 F1 (Churned)': round(f1_c1, 4)
    })

df_eval = pd.DataFrame(eval_rows)
print("Untouched Test Set Model Comparison:")
print(df_eval.to_string(index=False))

print("\n--- Step 7: Generating Figures ---")

# Figure 1: Dataset Splits and Class Priors Distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Subplot A: Split Sizes
splits_data = pd.DataFrame({
    'Split': ['Training (70%)', 'Validation (15%)', 'Test (15%)'],
    'Samples': [len(y_train), len(y_val), len(y_test)],
    'Stayed (0)': [(y_train == 0).sum(), (y_val == 0).sum(), (y_test == 0).sum()],
    'Churned (1)': [(y_train == 1).sum(), (y_val == 1).sum(), (y_test == 1).sum()]
})
x_pos = np.arange(len(splits_data))
width = 0.35
axes[0].bar(x_pos - width/2, splits_data['Stayed (0)'], width, label='Stayed (0)', color='#2b6cb0', alpha=0.9)
axes[0].bar(x_pos + width/2, splits_data['Churned (1)'], width, label='Churned (1)', color='#dd6b20', alpha=0.9)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(splits_data['Split'], fontweight='bold')
axes[0].set_ylabel('Number of Samples')
axes[0].set_title('A. Stratified Dataset Partitioning', pad=10, fontweight='bold')
axes[0].legend()
for i in range(len(splits_data)):
    axes[0].text(i - width/2, splits_data['Stayed (0)'][i] + 50, f"{splits_data['Stayed (0)'][i]}", ha='center', fontsize=9)
    axes[0].text(i + width/2, splits_data['Churned (1)'][i] + 50, f"{splits_data['Churned (1)'][i]}", ha='center', fontsize=9)

# Subplot B: Empirical Class Priors
priors_series = pd.Series({'Class 0 (Stayed)': prior_0, 'Class 1 (Churned)': prior_1})
colors_prior = ['#2b6cb0', '#dd6b20']
bars = axes[1].bar(priors_series.index, priors_series.values, color=colors_prior, width=0.45, alpha=0.9)
axes[1].set_ylabel('Empirical Prior Probability P(Y=k)')
axes[1].set_title('B. Empirical Training Class Priors', pad=10, fontweight='bold')
axes[1].set_ylim(0, 1.0)
for bar in bars:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f} ({yval*100:.1f}%)", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, "fig1_splits_and_priors.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"Saved {fig1_path}")

# Figure 2: The 20 Test Samples Class Probabilities
fig, ax = plt.subplots(figsize=(14, 6))
indices_20 = np.arange(20)
p0_vals = [rec['Posterior_P(Y=0|X)'] for rec in sample_20_records]
p1_vals = [rec['Posterior_P(Y=1|X)'] for rec in sample_20_records]
actuals = [rec['Actual_Class'] for rec in sample_20_records]
sample_labels = [f"S{rec['Sample_ID']}\n(Act:{rec['Actual_Class']})" for rec in sample_20_records]

w = 0.38
b1 = ax.bar(indices_20 - w/2, p0_vals, width=w, label='P(Y=0 | X) [Stayed]', color='#2b6cb0', alpha=0.9)
b2 = ax.bar(indices_20 + w/2, p1_vals, width=w, label='P(Y=1 | X) [Churned]', color='#dd6b20', alpha=0.9)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Decision Boundary (0.5)')

ax.set_xticks(indices_20)
ax.set_xticklabels(sample_labels, rotation=0, fontsize=8)
ax.set_ylabel('Model Estimated Probability')
ax.set_title('Model-Estimated Class Probabilities for 20 Representative Test Samples (GaussianNB)', pad=12, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right', frameon=True)

# Add dividing line between first 10 (actual 0) and next 10 (actual 1)
ax.axvline(9.5, color='red', linestyle=':', linewidth=1.5)
ax.text(4.5, 1.05, 'Actual Class: 0 (Stayed)', ha='center', color='#2b6cb0', fontweight='bold', fontsize=10)
ax.text(14.5, 1.05, 'Actual Class: 1 (Churned)', ha='center', color='#dd6b20', fontweight='bold', fontsize=10)

plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, "fig2_20_samples_probabilities.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"Saved {fig2_path}")

# Figure 3: Worked Bayes Feature Likelihood Ratios Bar Chart
fig, ax = plt.subplots(figsize=(10, 5))
log_ratios = [rec['Log_Lik_Y1'] - rec['Log_Lik_Y0'] for rec in worked_details]
feat_names_plot = [rec['Feature'] for rec in worked_details]
colors_ratio = ['#dd6b20' if r > 0 else '#2b6cb0' for r in log_ratios]

y_pos = np.arange(len(feat_names_plot))
ax.barh(y_pos, log_ratios, color=colors_ratio, alpha=0.9, height=0.6)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_yticks(y_pos)
ax.set_yticklabels(feat_names_plot, fontsize=9.5)
ax.set_xlabel('Log Likelihood Ratio: log[ p(x_j | Y=1) / p(x_j | Y=0) ]', fontweight='bold')
ax.set_title(f'Feature Evidence Contributions for Sample {worked_orig_id} (Actual: Churned)', pad=14, fontweight='bold', fontsize=12)
ax.set_xlim(-0.8, 2.0)

# Add clear legend or non-overlapping text callouts
ax.plot([], [], color='#dd6b20', linewidth=8, label='Pulls towards Churn (Y=1, LR > 1)')
ax.plot([], [], color='#2b6cb0', linewidth=8, label='Pulls towards Stay (Y=0, LR < 1)')
ax.legend(loc='lower right', frameon=True, fontsize=9)

plt.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, "fig3_worked_example_likelihoods.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"Saved {fig3_path}")

# Figure 4: Validation Tuning Curves for Decision Tree and Logistic Regression
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Decision Tree Tuning
axes[0].plot(tree_depths, train_f1_trees, marker='o', label='Train Macro F1', color='#4a5568', linestyle='--')
axes[0].plot(tree_depths, val_f1_trees, marker='s', label='Validation Macro F1', color='#2b6cb0', linewidth=2)
axes[0].axvline(best_depth, color='#dd6b20', linestyle=':', label=f'Optimal Depth = {best_depth}')
axes[0].set_xlabel('Tree Maximum Depth (max_depth)')
axes[0].set_ylabel('Macro F1 Score')
axes[0].set_title('A. Decision Tree Validation Tuning', pad=10, fontweight='bold')
axes[0].legend()

# Logistic Regression Tuning
c_labels = [str(c) for c in C_values]
axes[1].plot(range(len(C_values)), train_f1_lr, marker='o', label='Train Macro F1', color='#4a5568', linestyle='--')
axes[1].plot(range(len(C_values)), val_f1_lr, marker='s', label='Validation Macro F1', color='#2b6cb0', linewidth=2)
axes[1].set_xticks(range(len(C_values)))
axes[1].set_xticklabels(c_labels)
axes[1].axvline(C_values.index(best_C), color='#dd6b20', linestyle=':', label=f'Optimal C = {best_C}')
axes[1].set_xlabel('Inverse Regularization Strength (C)')
axes[1].set_ylabel('Macro F1 Score')
axes[1].set_title('B. Logistic Regression Validation Tuning', pad=10, fontweight='bold')
axes[1].legend()

plt.tight_layout()
fig4_path = os.path.join(FIGURES_DIR, "fig4_validation_tuning.png")
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"Saved {fig4_path}")

# Figure 5: Confusion Matrices for All 4 Models on Untouched Test Set
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

model_items = list(models.items())
class_names = ['Stayed (0)', 'Churned (1)']

for i, (name, preds) in enumerate(model_items):
    cm = confusion_matrix(y_test, preds)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    annot_labels = np.array([
        [f"{cm[0,0]}\n({cm_pct[0,0]:.1f}%)", f"{cm[0,1]}\n({cm_pct[0,1]:.1f}%)"],
        [f"{cm[1,0]}\n({cm_pct[1,0]:.1f}%)", f"{cm[1,1]}\n({cm_pct[1,1]:.1f}%)"]
    ])
    
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', ax=axes[i], cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 12, 'fontweight': 'bold'})
    axes[i].set_title(f"{name}\nAcc: {accuracy_score(y_test, preds):.3f} | Macro F1: {f1_score(y_test, preds, average='macro'):.3f}",
                      fontweight='bold', pad=8)
    axes[i].set_xlabel('Predicted Label', fontweight='bold')
    axes[i].set_ylabel('True Label', fontweight='bold')

plt.tight_layout()
fig5_path = os.path.join(FIGURES_DIR, "fig5_confusion_matrices.png")
plt.savefig(fig5_path, dpi=300)
plt.close()
print(f"Saved {fig5_path}")

# Figure 6: Model Comparison Bar Chart across Metrics
fig, ax = plt.subplots(figsize=(11, 5.5))
metrics = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
x = np.arange(len(metrics))
width = 0.2

for i, row in df_eval.iterrows():
    vals = [row['Accuracy'], row['Macro Precision'], row['Macro Recall'], row['Macro F1']]
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, vals, width, label=row['Model'])
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.2f}",
                ha='center', va='bottom', fontsize=8, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight='bold', fontsize=11)
ax.set_ylabel('Score (0.0 to 1.0)')
ax.set_title('Untouched Test Set Performance Comparison Across All Models', pad=12, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right', frameon=True)

plt.tight_layout()
fig6_path = os.path.join(FIGURES_DIR, "fig6_model_comparison.png")
plt.savefig(fig6_path, dpi=300)
plt.close()
print(f"Saved {fig6_path}")

# Export summary JSON for report generation
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

summary_data = {
    'train_samples': int(len(y_train)),
    'val_samples': int(len(y_val)),
    'test_samples': int(len(y_test)),
    'prior_0': float(prior_0),
    'prior_1': float(prior_1),
    'train_c0': int(N_0),
    'train_c1': int(N_1),
    'val_c0': int((y_val == 0).sum()),
    'val_c1': int((y_val == 1).sum()),
    'test_c0': int((y_test == 0).sum()),
    'test_c1': int((y_test == 1).sum()),
    'best_tree_depth': int(best_depth),
    'best_tree_val_f1': float(max(val_f1_trees)),
    'best_lr_C': float(best_C),
    'best_lr_val_f1': float(max(val_f1_lr)),
    'eval_table': eval_rows,
    'worked_example': {
        'orig_id': int(worked_orig_id),
        'actual': int(worked_raw['Exited']),
        'pred': int(preds_test[worked_idx_pos]),
        'prior_0': float(prior_0),
        'prior_1': float(prior_1),
        'post_0': float(post_test[worked_idx_pos, 0]),
        'post_1': float(post_test[worked_idx_pos, 1]),
        'raw_features': {col: (int(worked_raw[col]) if isinstance(worked_raw[col], (np.integer, int)) else (float(worked_raw[col]) if isinstance(worked_raw[col], (np.floating, float)) else str(worked_raw[col]))) for col in df_raw.columns if col not in drop_cols},
        'details': worked_details
    },
    'sample_20_records': sample_20_records
}

with open(os.path.join(BASE_DIR, "summary_data.json"), "w") as f:
    json.dump(summary_data, f, indent=2, cls=NpEncoder)

print("\n--- Asset Generation Complete! ---")
