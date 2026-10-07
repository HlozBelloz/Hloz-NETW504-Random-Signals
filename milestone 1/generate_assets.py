"""
NETW504 - Random Signals and Noise
Milestone 1 - Python Asset Generation Script
Student: Mazen Mohamed Hamdy Altelbany (ID: 64-12371)
Team: Hloz

This script runs the data analysis, generates simple clear plots,
and creates the cleaned CSV file.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
base_dir = r"e:\University\random signals\milestone 1"
data_path = r"e:\University\random signals\data set\Churn_Modelling.csv"
figures_dir = os.path.join(base_dir, "figures")

# Make sure figures folder exists
os.makedirs(figures_dir, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(data_path)

# Set simple clean plot style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})

# -------------------------------------------------------------
# Plot 1: Target Distribution
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
target_counts = df['Exited'].value_counts()
# Draw bar plot for 0 (Stayed) vs 1 (Left)
ax = sns.barplot(x=['0 (Stayed)', '1 (Exited)'], y=target_counts.values, palette=['#3274a1', '#e1812c'])
plt.title("Target Variable Distribution (Exited)")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")

# Add count and percentage text labels on bars
for i, count in enumerate(target_counts.values):
    pct = (count / len(df)) * 100
    ax.text(i, count / 2, f"{count}\n({pct:.1f}%)", ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "target_distribution.png"), dpi=300)
plt.close()

# -------------------------------------------------------------
# Plot 2: Missing Values Bar Plot
# -------------------------------------------------------------
plt.figure(figsize=(9, 4))
missing_counts = df.isnull().sum()
ax = sns.barplot(x=missing_counts.index, y=missing_counts.values, color='#3274a1')
plt.title("Missing Values per Column (Before Cleaning)")
plt.xlabel("Columns")
plt.ylabel("Missing Count")
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 5)

# Add text label above each bar
for i, count in enumerate(missing_counts.values):
    ax.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "missing_values.png"), dpi=300)
plt.close()

# List of the 8 numerical features
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# -------------------------------------------------------------
# Plot 3: Estimated PDF for each numerical feature
# -------------------------------------------------------------
fig, axes = plt.subplots(4, 2, figsize=(11, 13))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    # Normalized histogram with density=True and smooth KDE line
    sns.histplot(df[col], kde=True, stat="density", ax=axes[i], color='#3274a1', alpha=0.6)
    axes[i].set_title(f"Estimated PDF: {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Density")

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "numerical_pdfs.png"), dpi=300)
plt.close()

# -------------------------------------------------------------
# Plot 4: Empirical CDF for each numerical feature
# -------------------------------------------------------------
fig, axes = plt.subplots(4, 2, figsize=(11, 13))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    # Sort values to calculate empirical CDF: F(x) = rank / N
    sorted_vals = np.sort(df[col])
    y_vals = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    
    # Step plot for CDF
    axes[i].step(sorted_vals, y_vals, where='post', color='#2ca02c', linewidth=2)
    axes[i].set_title(f"Empirical CDF: {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("F(x)")
    axes[i].set_ylim(-0.02, 1.02)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "numerical_cdfs.png"), dpi=300)
plt.close()

# -------------------------------------------------------------
# Preprocessing Pipeline (Follow 4 Steps)
# -------------------------------------------------------------
# Step 1: Remove duplicate rows
cleaned_df = df.drop_duplicates().copy()

# Step 2: Remove rows where target is missing
cleaned_df = cleaned_df.dropna(subset=['Exited']).copy()

# Step 3: Fill missing numerical values with median
for col in num_cols:
    if cleaned_df[col].isnull().sum() > 0:
        median_val = cleaned_df[col].median()
        cleaned_df[col] = cleaned_df[col].fillna(median_val)

# Remove ID-like columns (RowNumber, CustomerId, Surname)
cleaned_df = cleaned_df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Step 4: One-hot encode categorical features and encode target
cleaned_df['Exited'] = cleaned_df['Exited'].astype(int)
cleaned_df = pd.get_dummies(cleaned_df, columns=['Geography', 'Gender'], dtype=int)

# Save cleaned CSV
cleaned_csv_path = os.path.join(base_dir, "Hloz_M1_Cleaned.csv")
cleaned_df.to_csv(cleaned_csv_path, index=False)
print("Saved cleaned CSV:", cleaned_csv_path)

# -------------------------------------------------------------
# Plot 5: Correlation Heatmap (Post-Cleaning)
# -------------------------------------------------------------
plt.figure(figsize=(8.5, 6.5))
corr_matrix = cleaned_df[num_cols].corr()

# Draw heatmap with numbers
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "correlation_heatmap.png"), dpi=300)
plt.close()

# -------------------------------------------------------------
# Plot 6: Simple Boxplots for Key Features vs Exited
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(data=df, x='Exited', y='Age', ax=axes[0], palette=['#3274a1', '#e1812c'])
axes[0].set_title("Age vs Churn (Exited)")
axes[0].set_xlabel("Exited (0: Stayed, 1: Left)")

sns.boxplot(data=df, x='Exited', y='CreditScore', ax=axes[1], palette=['#3274a1', '#e1812c'])
axes[1].set_title("CreditScore vs Churn (Exited)")
axes[1].set_xlabel("Exited (0: Stayed, 1: Left)")

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "boxplots_key_features.png"), dpi=300)
plt.close()

print("All figures and cleaned CSV created successfully.")
