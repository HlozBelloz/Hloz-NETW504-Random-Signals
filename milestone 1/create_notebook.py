"""
Script to create Hloz_M1.ipynb with simple, clean student code,
line-by-line comments, and clear observations.
"""

import os
import nbformat as nbf
from nbclient import NotebookClient

notebook_path = r"e:\University\random signals\milestone 1\Hloz_M1.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.12.0"
    }
}

cells = []

# Title cell
cells.append(nbf.v4.new_markdown_cell("""# NETW504 - Random Signals and Noise
## Milestone 1: Exploratory Data Analysis (EDA)
- **Instructor:** Prof. Talal Elshabrawy
- **Team Name:** Hloz
- **Team Members:**
  - Mazen Mohamed Hamdy Altelbany (ID: 64-12371)
  - Malak Sherif Mohamed (ID: 64-10784)
  - Suhad Eyhab Rasheed (ID: 64-31506)
  - Sama Ismael Ahel (ID: 64-19880)

---
### Goal of this Milestone:
Perform basic Exploratory Data Analysis (EDA) on our approved dataset (Bank Customer Churn), visualize distributions (PDF and CDF), check data quality, and apply basic preprocessing steps.
"""))

# Section 1: Dataset and Target
cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset and Target

- **Dataset Name:** Bank Customer Churn Prediction
- **Kaggle Link:** [https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)
- **Row Meaning:** Each row represents one bank customer, including their personal demographics, account balance, credit info, and whether they stayed with the bank or left.
- **Target Column:** `Exited` (1 = Customer left the bank, 0 = Customer stayed with the bank).
"""))

# Cell 1: Load Data and Check Target
cells.append(nbf.v4.new_code_cell("""# Import the required libraries
import pandas as pd            # Used for data manipulation and loading CSV files
import numpy as np             # Used for numerical calculations and arrays
import matplotlib.pyplot as plt # Used for plotting graphs
import seaborn as sns          # Used for styled data visualization

# Set a clean plot style
sns.set_theme(style="whitegrid")

# Load the dataset from the CSV file
df = pd.read_csv('../data set/Churn_Modelling.csv')

# Print the size of the dataset (rows and columns)
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Count how many customers stayed (0) and how many left (1)
target_counts = df['Exited'].value_counts()

# Calculate the percentage of each class
target_percentages = df['Exited'].value_counts(normalize=True) * 100

# Put them together into a neat summary table
target_summary = pd.DataFrame({
    'Count': target_counts,
    'Percentage (%)': target_percentages.round(2)
})

# Display the summary table
print("\\nTarget Class Summary:")
display(target_summary)
"""))

# Section 1.2: Feature Types
cells.append(nbf.v4.new_markdown_cell("""### Feature Types and ID Columns

- **ID-like Columns (To be removed later):**
  - `RowNumber`: Just the row index.
  - `CustomerId`: Random customer ID number.
  - `Surname`: Customer last name (free text).
- **Numerical Features (8):**
  - `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.
- **Categorical Features (2):**
  - `Geography` (Country: France, Spain, Germany)
  - `Gender` (Female, Male)
"""))

# Cell 2: Feature Types Summary Table
cells.append(nbf.v4.new_code_cell("""# Create a table showing each column and its type
feature_table = pd.DataFrame({
    'Column Name': df.columns,
    'Data Type': [df[col].dtype for col in df.columns],
    'Feature Type': [
        'ID Column (Drop)', 'ID Column (Drop)', 'ID Column (Drop)',
        'Numerical', 'Categorical', 'Categorical',
        'Numerical', 'Numerical', 'Numerical',
        'Numerical', 'Numerical (Binary)', 'Numerical (Binary)',
        'Numerical', 'Target'
    ]
})

# Display the feature classification table
display(feature_table)
"""))

# Section 2: Initial Inspection and Data Quality
cells.append(nbf.v4.new_markdown_cell("""## 2. Initial Inspection and Data Quality

Here we look at the first 5 rows, check for duplicate rows, check for missing values, and print numerical statistics.
"""))

# Cell 3: First 5 rows and info
cells.append(nbf.v4.new_code_cell("""# Display the first 5 rows of the dataset
print("First 5 rows:")
display(df.head())

# Show column data types and non-null counts
print("\\nDataset Info:")
df.info()
"""))

# Cell 4: Duplicate rows and missing values
cells.append(nbf.v4.new_code_cell("""# Check for exact duplicate rows
num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")

# Count missing (null) values in every column
missing_counts = df.isnull().sum()

# Calculate missing value percentages
missing_percent = (missing_counts / len(df)) * 100

# Put missing values into a table
missing_table = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing (%)': missing_percent
})

# Display missing values table
print("\\nMissing Values per Column:")
display(missing_table)
"""))

# Cell 5: Descriptive Statistics for Numerical Features
cells.append(nbf.v4.new_markdown_cell("""### Numerical Features Statistics
We calculate: `count`, `mean`, `standard deviation`, `min`, `median`, and `max`.
"""))

cells.append(nbf.v4.new_code_cell("""# List of the 8 numerical input features
numerical_features = [
    'CreditScore', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

# Calculate the required summary statistics
stats_table = pd.DataFrame({
    'count': df[numerical_features].count(),
    'mean': df[numerical_features].mean().round(2),
    'std': df[numerical_features].std().round(2),
    'min': df[numerical_features].min().round(2),
    'median': df[numerical_features].median().round(2),
    'max': df[numerical_features].max().round(2)
})

# Display the statistics table
print("Numerical Features Descriptive Statistics:")
display(stats_table)
"""))

# Quality Comments
cells.append(nbf.v4.new_markdown_cell("""### Comments on Data Quality:
- **Class Balance:** The target is moderately imbalanced. Around 79.6% stayed (`0`) and 20.4% exited (`1`).
- **Duplicates and Missing Values:** There are 0 duplicate rows and 0 missing values in any column. The data is complete and clean.
- **Unusual Values:** Over 36% of customers have an account balance of 0, which makes the `Balance` distribution bimodal. `Age` has a slight right skew with some older customers up to 92 years old.
"""))

# Section 3: Required EDA Plots
cells.append(nbf.v4.new_markdown_cell("""## 3. Required EDA Plots
Every plot group below has a clear title, labeled axes, and a 1–3 sentence observation.
"""))

# Plot 1: Target distribution
cells.append(nbf.v4.new_markdown_cell("""### 3.1 Target Distribution Plot"""))
cells.append(nbf.v4.new_code_cell("""# Create a figure for the target distribution
plt.figure(figsize=(6, 4))

# Plot a bar chart for Exited (0 vs 1)
ax = sns.barplot(x=['0 (Stayed)', '1 (Exited)'], y=target_counts.values, palette=['#3274a1', '#e1812c'])

# Set plot title and axis labels
plt.title("Target Distribution (Exited)", fontsize=12, fontweight='bold')
plt.xlabel("Customer Status", fontsize=10)
plt.ylabel("Number of Customers", fontsize=10)

# Add text labels on top of the bars showing count and percentage
for i, count in enumerate(target_counts.values):
    pct = (count / len(df)) * 100
    ax.text(i, count + 100, f"{count} ({pct:.1f}%)", ha='center', fontsize=10, fontweight='bold')

# Give some extra space on y-axis
plt.ylim(0, 9000)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""> **Observation (Target Distribution):**  
> Around 79.6% of the customers stayed with the bank while 20.4% left. This shows an imbalanced dataset (~4 to 1 ratio), so accuracy alone will not be enough to evaluate our future models.
"""))

# Plot 2: Missing Values
cells.append(nbf.v4.new_markdown_cell("""### 3.2 Missing Values Plot"""))
cells.append(nbf.v4.new_code_cell("""# Create a figure for the missing values
plt.figure(figsize=(9, 4))

# Bar plot of missing counts across all columns
ax = sns.barplot(x=missing_counts.index, y=missing_counts.values, color='#3274a1')

# Set titles and labels
plt.title("Missing Values per Column (Before Cleaning)", fontsize=12, fontweight='bold')
plt.xlabel("Columns", fontsize=10)
plt.ylabel("Missing Count", fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 5)

# Show count above each bar
for i, count in enumerate(missing_counts.values):
    ax.text(i, count + 0.1, str(count), ha='center', fontsize=9)

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""> **Observation (Missing Values):**  
> All columns in the dataset have exactly zero missing values. This means no rows or columns need to be dropped or imputed due to missing data.
"""))

# Plot 3: Numerical Features - Estimated PDF
cells.append(nbf.v4.new_markdown_cell("""### 3.3 Numerical Features – Estimated PDF
Normalized histogram (`stat="density"`) with a KDE curve for each of the 8 numerical features.
"""))

cells.append(nbf.v4.new_code_cell("""# Create a grid of 4 rows and 2 columns for the 8 features
fig, axes = plt.subplots(4, 2, figsize=(12, 14))
axes = axes.flatten()

# Loop through each numerical feature and plot its PDF
for i, col in enumerate(numerical_features):
    # Plot histogram with normalized density and smooth KDE line
    sns.histplot(df[col], kde=True, stat="density", ax=axes[i], color='#3274a1', alpha=0.55)
    
    # Set titles and axis labels
    axes[i].set_title(f"Estimated PDF: {col}", fontsize=11, fontweight='bold')
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel("Probability Density", fontsize=10)

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""> **Observation (Numerical PDFs):**  
> `CreditScore` looks mostly normal (bell-shaped) centered at 650, while `Age` is right-skewed with most customers between 30 and 40. `Balance` has a large spike at zero because many customers have no balance, and `EstimatedSalary` is flat, looking like a uniform distribution.
"""))

# Plot 4: Numerical Features - Empirical CDF
cells.append(nbf.v4.new_markdown_cell("""### 3.4 Numerical Features – Empirical CDF
Empirical Cumulative Distribution Function showing cumulative probability $F_X(x) = P(X \\le x)$.
"""))

cells.append(nbf.v4.new_code_cell("""# Create a grid of 4 rows and 2 columns for the CDFs
fig, axes = plt.subplots(4, 2, figsize=(12, 14))
axes = axes.flatten()

# Loop through each numerical feature and plot its empirical CDF
for i, col in enumerate(numerical_features):
    # Sort the data values from smallest to largest
    sorted_values = np.sort(df[col])
    
    # Calculate cumulative probability: 1/N, 2/N, ..., N/N
    cumulative_prob = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    
    # Step plot for empirical CDF
    axes[i].step(sorted_values, cumulative_prob, where='post', color='#2ca02c', linewidth=2)
    
    # Set titles and axis labels
    axes[i].set_title(f"Empirical CDF: {col}", fontsize=11, fontweight='bold')
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel("Cumulative Probability F(x)", fontsize=10)
    axes[i].set_ylim(-0.02, 1.02)
    axes[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""> **Observation (Empirical CDFs):**  
> The CDF for `Balance` shows an immediate vertical jump at zero up to ~36%, proving that more than a third of accounts have zero balance. `EstimatedSalary` forms a straight diagonal line, which is typical for a uniform distribution, while `CreditScore` and `Age` show smooth S-curves.
"""))

# Section 4: Basic Preprocessing
cells.append(nbf.v4.new_markdown_cell("""## 4. Basic Preprocessing Pipeline (Follow Strict Rules)

We follow the 4 required steps in order:
1. Remove exact duplicate rows.
2. Remove rows where the target is missing.
3. Impute missing numerical values using the median.
4. One-hot encode categorical features and encode the target labels as integers.

*Note: We keep `df` as the original dataset and store the cleaned version in `cleaned_df`.*
"""))

# Cell 6: Run Preprocessing Steps
cells.append(nbf.v4.new_code_cell("""# Make a copy of the dataset so the original remains untouched
cleaned_df = df.copy()

# Step 1: Remove exact duplicate rows
before_dup = len(cleaned_df)
cleaned_df = cleaned_df.drop_duplicates()
print(f"Step 1: Removed {before_dup - len(cleaned_df)} duplicate rows.")

# Step 2: Remove rows where the target column (Exited) is missing
before_na = len(cleaned_df)
cleaned_df = cleaned_df.dropna(subset=['Exited'])
print(f"Step 2: Removed {before_na - len(cleaned_df)} rows with missing target.")

# Step 3: Fill missing numerical values with the feature median
imputed_count = 0
for col in numerical_features:
    if cleaned_df[col].isnull().sum() > 0:
        median_val = cleaned_df[col].median()
        cleaned_df[col] = cleaned_df[col].fillna(median_val)
        imputed_count += 1
print(f"Step 3: Missing numerical values filled using median. Features imputed: {imputed_count}")

# Drop ID-like columns that are not useful for machine learning
id_cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
cleaned_df = cleaned_df.drop(columns=id_cols_to_drop)
print(f"Dropped ID columns: {id_cols_to_drop}")

# Step 4: One-hot encode categorical input features and encode target
# Target is already 0 and 1, ensure integer type
cleaned_df['Exited'] = cleaned_df['Exited'].astype(int)

# One-hot encode Geography and Gender
categorical_cols = ['Geography', 'Gender']
cleaned_df = pd.get_dummies(cleaned_df, columns=categorical_cols, dtype=int)

# Print the encoding mapping dictionary
print("\\nEncoding Mapping:")
print("- Target 'Exited': 0 = Stayed (Retained), 1 = Left (Churned)")
print("- Geography -> One-hot encoded into Geography_France, Geography_Germany, Geography_Spain")
print("- Gender -> One-hot encoded into Gender_Female, Gender_Male")

# Print final shape
print(f"\\nFinal Cleaned Dataset Shape: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")

# Display first 5 rows of the cleaned data
display(cleaned_df.head())

# Save cleaned dataset to CSV
cleaned_df.to_csv("Hloz_M1_Cleaned.csv", index=False)
print("Saved cleaned data to 'Hloz_M1_Cleaned.csv'!")
"""))

# Plot 5: Correlation Heatmap
cells.append(nbf.v4.new_markdown_cell("""### 3.5 Correlation Heatmap (After Preprocessing)
Correlation matrix and heatmap for the numerical input features after preprocessing.
"""))

cells.append(nbf.v4.new_code_cell("""# Calculate Pearson correlation between numerical features
plt.figure(figsize=(9, 7))
corr_matrix = cleaned_df[numerical_features].corr()

# Draw heatmap with correlation values rounded to 2 decimal places
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)

# Set title
plt.title("Correlation Heatmap of Numerical Features (Post-Cleaning)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""> **Observation (Correlation Heatmap):**  
> Most numerical features have very low correlation with each other (close to 0). The only noticeable correlation is between `Balance` and `NumOfProducts` (-0.30), meaning customers with more products tend to have lower account balances. Since correlations are low, multicollinearity won't be an issue for model training.
"""))

# Section 5: Conclusion
cells.append(nbf.v4.new_markdown_cell("""## 5. Milestone 1 Conclusion (Summary of Learnings)
- **Data Quality:** The dataset is complete with 10,000 rows, zero duplicate rows, and zero missing values across all columns.
- **Target Imbalance:** About 79.6% of customers stayed and 20.4% left, showing a moderate class imbalance.
- **Distributions:** `CreditScore` is bell-shaped, `Age` is skewed towards older customers, `EstimatedSalary` is uniform, and `Balance` has a large spike at 0 (zero-inflated).
- **Preprocessing:** We removed 3 useless ID columns (`RowNumber`, `CustomerId`, `Surname`) and one-hot encoded `Geography` and `Gender`.
- **Output:** The final cleaned dataset `Hloz_M1_Cleaned.csv` has 10,000 rows and 14 clean numerical columns, ready for Milestone 2.
"""))

nb.cells = cells

# Save notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook created at: {notebook_path}")

# Execute notebook to pre-render all outputs
print("Executing notebook to pre-render outputs...")
client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook executed and saved successfully!")
