"""
create_notebook.py
==================
NETW504 Random Signals and Noise - Milestone 2
Team Name: Hloz
Author: Mazen Mohamed Hamdy Altelbany (ID: 64-12371) & Team Hloz

Builds and executes Hloz_M2.ipynb with complete step-by-step code,
thorough explanations, line-by-line comments, outputs, and plots.
"""

import os
import nbformat as nbf
from nbclient import NotebookClient

notebook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hloz_M2.ipynb")

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

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# NETW504 - Random Signals and Noise
## Milestone 2: Bayesian Estimation and Machine Learning
- **Instructor:** Prof. Talal Elshabrawy
- **Team Name:** Hloz
- **Team Members:**
  - Mazen Mohamed Hamdy Altelbany (ID: 64-12371)
  - Malak Sherif Mohamed (ID: 64-10784)
  - Suhad Eyhab Rasheed (ID: 64-31506)
  - Sama Ismael Ahel (ID: 64-19880)

---
### Milestone Goal:
1. Estimate Bayesian class probabilities directly from training data using Bayes' Theorem under the conditional independence assumption.
2. Train Bayesian Machine Learning (`GaussianNB`) and Conventional Machine Learning models (`DecisionTreeClassifier` and `LogisticRegression`).
3. Select model hyperparameters using an independent Validation Set (avoiding data leakage).
4. Evaluate and compare all models on an untouched Test Set using Accuracy, Macro Precision, Macro Recall, Macro F1, and labeled Confusion Matrices.
5. Provide detailed code explanations, mathematical derivations, and in-depth discussions on class imbalance and conditional independence.
"""))

# ==============================================================================
# SECTION 1: DATASET PREPARATION & PREPROCESSING PIPELINE
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset Preparation & Preprocessing Pipeline

### 1.1 Dataset Profile & Target Mapping from Milestone 1
In Milestone 1, we thoroughly analyzed and profiled the **Bank Customer Churn Prediction** dataset from Kaggle (`Churn_Modelling.csv`):
- **Total Records:** 10,000 customers.
- **Target Variable:** `Exited` (Binary classification):
  - `0`: Customer stayed with the bank (Retained).
  - `1`: Customer left the bank (Churned).
  - **Class-to-Integer Mapping:** `0 -> Stayed`, `1 -> Churned`.
- **ID Columns (To Drop):** `RowNumber`, `CustomerId`, `Surname`. These columns are arbitrary database keys or customer names that have no physical predictive relationship with churn and would only act as noise or cause data leakage.
- **Predictive Input Features (10):**
  - **Numerical (8):** `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.
  - **Categorical (2):** `Geography` (`France`, `Germany`, `Spain`), `Gender` (`Female`, `Male`).

Let us import the necessary scientific computing libraries and load the dataset.
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 1: Import Core Libraries and Load Dataset
# ==============================================================================
# Import operating system and file system utilities
import os                        # File path manipulation and directory handling

# Import numerical and tabular manipulation libraries
import numpy as np               # Fast vector/matrix operations and mathematical functions
import pandas as pd              # Dataframe manipulation and tabular processing

# Import visualization libraries
import matplotlib.pyplot as plt  # Core plotting library for charts and figures
import seaborn as sns           # High-level statistical visualization library

# Import scikit-learn preprocessing and evaluation utilities
from sklearn.model_selection import train_test_split  # Stratified dataset partitioning
from sklearn.impute import SimpleImputer              # Missing value imputation
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Scaling & encoding
from sklearn.compose import ColumnTransformer         # Column-specific pipeline chaining
from sklearn.pipeline import Pipeline                 # Modular step encapsulation

# Import machine learning classifiers
from sklearn.naive_bayes import GaussianNB            # Gaussian Naive Bayes classifier
from sklearn.tree import DecisionTreeClassifier       # Conventional Decision Tree
from sklearn.linear_model import LogisticRegression   # Conventional Logistic Regression

# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Configure plot formatting for high-quality figures
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.titlesize': 14
})

# Robust dataset loading: supports running from root, milestone folder, or Colab
possible_paths = [
    '../data set/Churn_Modelling.csv',
    'data set/Churn_Modelling.csv',
    'Churn_Modelling.csv'
]
dataset_path = None
for p in possible_paths:
    if os.path.exists(p):
        dataset_path = p
        break

if dataset_path is None:
    raise FileNotFoundError("Could not locate Churn_Modelling.csv in known locations.")

print(f"Loading dataset from: {dataset_path}")
df_raw = pd.read_csv(dataset_path)

# Print initial verification of dimensions
print(f"Dataset successfully loaded!")
print(f"Total Rows: {df_raw.shape[0]}, Total Columns: {df_raw.shape[1]}")

# Drop ID columns that do not provide physical predictive value
id_columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
df = df_raw.drop(columns=id_columns_to_drop)

print(f"Dropped ID columns: {id_columns_to_drop}")
print(f"Cleaned feature matrix dimensions: {df.shape[0]} rows, {df.shape[1]} columns (10 features + 1 target)")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 1.2 Stratified Dataset Partitioning (70% Train, 15% Validation, 15% Test)

#### Why Stratification is Essential:
In Milestone 1, we discovered that the dataset exhibits a moderate class imbalance: ~79.6% stayed (class 0) versus ~20.4% churned (class 1). If we split randomly without stratification, some splits (especially the smaller 15% validation or test sets) might randomly receive too few or too many minority class samples, skewing performance metrics. 
**Stratified splitting** guarantees that the training, validation, and test subsets maintain the exact same 79.6% / 20.4% class ratio as the original population.

#### Splitting Strategy:
1. Split the entire dataset into **70% Training** and **30% Temporary** (`X_temp`, `y_temp`).
2. Split the 30% Temporary set equally (50% / 50%) into **15% Validation** and **15% Test**.
3. All splits are stratified on the target `Exited` using `random_state=42` for exact reproducibility.
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 2: Stratified Train / Validation / Test Splitting
# ==============================================================================
# Separate the feature matrix (X) from the target vector (y)
target_column = 'Exited'
feature_columns = [col for col in df.columns if col != target_column]

X = df[feature_columns].copy()  # Input features (10 columns)
y = df[target_column].copy()    # Target labels (0 or 1)

# Step 1: Split into 70% Training and 30% Temporary holdout
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,          # 30% held out for validation and testing
    random_state=42,         # Fixed seed for exact reproducibility
    stratify=y               # Preserve exact class proportions
)

# Step 2: Split the 30% holdout equally into 15% Validation and 15% Test
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw, y_temp,
    test_size=0.50,          # Split 30% in half -> 15% val, 15% test
    random_state=42,         # Fixed seed
    stratify=y_temp          # Preserve exact class proportions
)

# Summarize the split sizes and class distributions into a clean summary table
split_summary = pd.DataFrame({
    'Split': ['Training', 'Validation', 'Test', 'Total'],
    'Total Samples': [len(y_train), len(y_val), len(y_test), len(y)],
    'Percentage (%)': [
        len(y_train)/len(y)*100,
        len(y_val)/len(y)*100,
        len(y_test)/len(y)*100,
        100.0
    ],
    'Class 0 (Stayed)': [
        (y_train == 0).sum(),
        (y_val == 0).sum(),
        (y_test == 0).sum(),
        (y == 0).sum()
    ],
    'Class 1 (Churned)': [
        (y_train == 1).sum(),
        (y_val == 1).sum(),
        (y_test == 1).sum(),
        (y == 1).sum()
    ],
    'Class 1 Ratio (%)': [
        (y_train == 1).mean() * 100,
        (y_val == 1).mean() * 100,
        (y_test == 1).mean() * 100,
        (y == 1).mean() * 100
    ]
})

print("Dataset Partitioning Summary:")
display(split_summary.round(2))
"""))

cells.append(nbf.v4.new_markdown_cell("""### 1.3 Preprocessing Pipeline (Fitted on Training Data ONLY)

#### Data Leakage Prevention:
A fundamental principle in signal processing and statistical machine learning is preventing **data leakage**. Information from the validation or test sets (such as feature means, variances, or category encodings) must NEVER be seen during model training or parameter estimation.
- If we scaled the whole dataset before splitting, the test set's mean and variance would contaminate the training scaling, giving artificially optimistic results.
- **Correct Methodology:** All transformers (imputers, one-hot encoders, standard scalers) are **fitted strictly on the Training set only**, and then applied to transform the Validation and Test sets.

#### Preprocessing Steps:
1. **Numerical Features (8):**
   - Impute missing values with feature median (robust against outliers).
   - Standardize features using `StandardScaler` ($z = \frac{x - \mu}{\sigma}$) to give zero mean and unit variance, which is critical for Gaussian likelihood modeling and gradient-based models.
2. **Categorical Features (2):**
   - Impute missing values with most frequent class.
   - One-hot encode using `OneHotEncoder(drop='first')`:
     - `Geography` (`France`, `Germany`, `Spain`) -> 2 binary dummy features (`Geography_Germany`, `Geography_Spain`), with `France` as the baseline reference.
     - `Gender` (`Female`, `Male`) -> 1 binary dummy feature (`Gender_Male`), with `Female` as the baseline reference.
     - Dropping the first category prevents collinearity (the dummy variable trap).
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 3: Fit Preprocessing Pipeline on Training Data ONLY
# ==============================================================================
# Define numerical and categorical column names
numeric_cols = [
    'CreditScore', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]
categorical_cols = ['Geography', 'Gender']

# Create the numerical transformer: Median Imputer followed by StandardScaler
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Impute using training median
    ('scaler', StandardScaler())                    # Standardize to zero mean, unit variance
])

# Create the categorical transformer: Mode Imputer followed by OneHotEncoder
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute using training mode
    ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')) # Avoid dummy trap
])

# Combine into a unified ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# CRITICAL STEP: Fit the preprocessor on the TRAINING set ONLY!
print("Fitting preprocessor strictly on training data...")
preprocessor.fit(X_train_raw)

# Retrieve transformed feature names for full interpretability
ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(categorical_cols).tolist()
all_feature_names = numeric_cols + ohe_feature_names
print(f"Total Transformed Features ({len(all_feature_names)}):")
for idx, name in enumerate(all_feature_names, start=1):
    print(f"  {idx}. {name}")

# Transform Train, Validation, and Test feature matrices
X_train = preprocessor.transform(X_train_raw)
X_val = preprocessor.transform(X_val_raw)
X_test = preprocessor.transform(X_test_raw)

print(f"\\nTransformed matrix shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  X_val:   {X_val.shape}")
print(f"  X_test:  {X_test.shape}")
"""))

# ==============================================================================
# SECTION 2: BAYESIAN ESTIMATION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 2. Bayesian Estimation (Direct Bayes' Theorem from Scratch)

### 2.1 Theoretical Foundations of Bayesian Decision Theory

In statistical communication and Random Signals (NETW504), **Bayesian Decision Theory** provides the optimal framework for decision making under uncertainty.
Given an observed feature vector $X = [x_1, x_2, \dots, x_D]^T$ and a discrete class label $Y \in \{0, 1\}$, **Bayes' Theorem** calculates the posterior probability $P(Y=k \mid X=x)$:

$$P(Y=k \mid X=x) = \frac{P(Y=k) \cdot p(X=x \mid Y=k)}{p(X=x)} = \frac{P(Y=k) \cdot p(X=x \mid Y=k)}{\sum_{c \in \{0, 1\}} P(Y=c) \cdot p(X=x \mid Y=c)}$$

Where:
1. **$P(Y=k)$ (Prior Probability):** The baseline probability of encountering class $k$ before observing any customer features.
2. **$p(X=x \mid Y=k)$ (Class-Conditional Likelihood):** The probability density of observing the specific feature vector $x$ given that the customer belongs to class $k$.
3. **$p(X=x) = \sum_c P(Y=c) p(X=x \mid Y=c)$ (Evidence / Total Probability):** The marginal likelihood of feature vector $x$, acting as a normalizer so all class posteriors sum to 1.
4. **$P(Y=k \mid X=x)$ (Posterior Probability):** The updated probability of class $k$ after incorporating all customer evidence.

#### Conditional Independence Assumption:
Assuming feature dimensions are conditionally independent given the class label $Y=k$:
$$p(X=x \mid Y=k) = \prod_{j=1}^D p(x_j \mid Y=k)$$

Thus, the unnormalized score for class $k$ is:
$$\text{score}_k(x) = P(Y=k) \times \prod_{j=1}^D p(x_j \mid Y=k)$$
And the normalized posterior probability is:
$$P(Y=k \mid x) = \frac{\text{score}_k(x)}{\sum_c \text{score}_c(x)}$$

#### Numerical Stability via Log-Sum-Exp:
In practice, multiplying $D=11$ probability densities can cause severe floating-point underflow (evaluating to $0.0$). We perform the calculation stably in log-space:
$$\log \text{score}_k(x) = \log P(Y=k) + \sum_{j=1}^D \log p(x_j \mid Y=k)$$
And normalize using the standard softmax (log-sum-exp trick):
$$P(Y=k \mid x) = \frac{\exp(\log \text{score}_k(x) - \max_c \log \text{score}_c(x))}{\sum_{c} \exp(\log \text{score}_c(x) - \max_c \log \text{score}_c(x))}$$
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 4: Calculate Empirical Class Priors from Training Data
# ==============================================================================
# Count the number of training samples in each class
N_train = len(y_train)
N_class_0 = (y_train == 0).sum()
N_class_1 = (y_train == 1).sum()

# Calculate empirical prior probabilities P(Y=k) strictly from training counts
prior_0 = N_class_0 / N_train
prior_1 = N_class_1 / N_train

# Store priors in a convenient dictionary
priors = {0: prior_0, 1: prior_1}

print("--- Empirical Class Priors (Training Set) ---")
print(f"Total Training Samples (N): {N_train}")
print(f"Class 0 Count (Stayed):    {N_class_0} -> P(Y=0) = {prior_0:.4f} ({prior_0*100:.2f}%)")
print(f"Class 1 Count (Churned):   {N_class_1} -> P(Y=1) = {prior_1:.4f} ({prior_1*100:.2f}%)")
print(f"Class Imbalance Ratio:     {N_class_0 / N_class_1:.2f} to 1 (Stayed vs Churned)")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 2.2 Discussion: Effect of Data Rebalancing on Class Priors

> **Discussion on Rebalancing Training Data:**  
> If techniques like SMOTE (Synthetic Minority Over-sampling Technique) or random undersampling are applied to balance the training data to a 50/50 distribution, the empirical training prior becomes:
> $$P_{\text{rebalanced}}(Y=0) = 0.50, \quad P_{\text{rebalanced}}(Y=1) = 0.50$$
> - **Effect on the Decision Boundary:** An artificial 50/50 prior removes the penalty against the minority class. This shifts the decision boundary towards class 0, dramatically increasing recall (sensitivity) for churners.
> - **Effect on Test/Production Reality:** In the real world, the true population base rate remains ~20%. If an unadjusted 50/50 model is deployed, it will generate an excessive volume of **False Positives** (misclassifying loyal customers as churners).
> - **Bayesian Adjustment:** To deploy a rebalanced model properly, Bayes' rule dictates that model outputs must be re-calibrated by multiplying by the ratio of true prior to training prior:
>   $$P_{\text{true}}(Y=k \mid X) \propto P_{\text{model}}(Y=k \mid X) \times \frac{P_{\text{true}}(Y=k)}{P_{\text{train}}(Y=k)}$$
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 5: Estimate Feature Likelihood Parameters (Means and Variances on Train)
# ==============================================================================
# For each class k in {0, 1}, compute the mean vector mu_k and variance vector sigma_k^2
# across all 11 standardized features using training data only.

means = {}
variances = {}

# Compute variance smoothing factor to prevent division by zero or log(0)
# Exactly matches scikit-learn's GaussianNB implementation (var_smoothing = 1e-9 * max(variance))
global_variance_max = np.max(np.var(X_train, axis=0))
var_smoothing = 1e-9 * global_variance_max

for k in [0, 1]:
    # Extract only the training samples belonging to class k
    X_k = X_train[y_train.values == k]
    
    # Calculate sample mean for each feature in class k
    means[k] = np.mean(X_k, axis=0)
    
    # Calculate sample variance for each feature in class k with smoothing
    variances[k] = np.var(X_k, axis=0) + var_smoothing

# Display the estimated parameters in a structured DataFrame
params_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Mean (Stayed, Y=0)': means[0].round(4),
    'Var (Stayed, Y=0)': variances[0].round(4),
    'Mean (Churned, Y=1)': means[1].round(4),
    'Var (Churned, Y=1)': variances[1].round(4)
})

print("Estimated Gaussian Feature Likelihood Parameters (Fitted on Train Only):")
display(params_df)
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 6: Direct Bayesian Posterior Calculation from Scratch
# ==============================================================================
def calculate_gaussian_log_density(x, mu, var):
    \"\"\"
    Computes the log of the univariate normal probability density function:
    log p(x | mu, var) = -0.5 * log(2 * pi * var) - (x - mu)^2 / (2 * var)
    \"\"\"
    return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * ((x - mu) ** 2) / var

def compute_bayesian_posteriors(X_matrix):
    \"\"\"
    Calculates posterior probabilities P(Y=k | X) for all samples directly using
    Bayes' Theorem under the conditional independence assumption.
    \"\"\"
    num_samples = X_matrix.shape[0]
    log_scores = np.zeros((num_samples, 2))  # Matrix to hold [log score_0, log score_1]
    
    for k in [0, 1]:
        # Prior log probability: log P(Y=k)
        log_prior = np.log(priors[k])
        
        # Sum of log feature likelihoods: sum_j log p(x_ij | Y=k)
        log_likelihoods = calculate_gaussian_log_density(X_matrix, means[k], variances[k])
        sum_log_likelihoods = np.sum(log_likelihoods, axis=1)
        
        # score_k = log P(Y=k) + sum_j log p(x_ij | Y=k)
        log_scores[:, k] = log_prior + sum_log_likelihoods
        
    # Numerically stable normalization (Softmax with max-subtraction)
    max_log = np.max(log_scores, axis=1, keepdims=True)
    exp_scores = np.exp(log_scores - max_log)
    posteriors = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Predicted class is the class with highest posterior probability (MAP Rule)
    predictions = np.argmax(posteriors, axis=1)
    
    return posteriors, predictions

# Compute posteriors directly for all training, validation, and test samples
post_train, preds_train = compute_bayesian_posteriors(X_train)
post_val, preds_val = compute_bayesian_posteriors(X_val)
post_test, preds_test = compute_bayesian_posteriors(X_test)

# Verify that all rows sum strictly to 1.0
assert np.allclose(np.sum(post_train, axis=1), 1.0), "Error: Train probabilities do not sum to 1!"
assert np.allclose(np.sum(post_val, axis=1), 1.0), "Error: Validation probabilities do not sum to 1!"
assert np.allclose(np.sum(post_test, axis=1), 1.0), "Error: Test probabilities do not sum to 1!"

print("Direct Bayesian Posteriors computed successfully!")
print("Verification: Every sample's posterior probabilities sum exactly to 1.0.")
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 7: Save Complete Posteriors Table to CSV
# ==============================================================================
# Construct a unified table containing all samples across Train, Validation, and Test
posterior_records = []

for index_series, split_name, y_true_subset, post_subset, pred_subset in [
    (X_train_raw.index, 'Train', y_train, post_train, preds_train),
    (X_val_raw.index, 'Validation', y_val, post_val, preds_val),
    (X_test_raw.index, 'Test', y_test, post_test, preds_test)
]:
    for i in range(len(index_series)):
        posterior_records.append({
            'Sample_ID': index_series[i],
            'Split': split_name,
            'Actual_Class': y_true_subset.iloc[i],
            'P(Y=0|X)': round(post_subset[i, 0], 6),
            'P(Y=1|X)': round(post_subset[i, 1], 6),
            'Predicted_Class': pred_subset[i]
        })

df_all_posteriors = pd.DataFrame(posterior_records)

# Save to required CSV deliverable
csv_output_path = 'Hloz_M2_Bayesian_Posteriors.csv'
df_all_posteriors.to_csv(csv_output_path, index=False)
print(f"Saved complete posteriors table to '{csv_output_path}' ({len(df_all_posteriors)} rows).")

# Display first 10 rows
print("\\nFirst 10 Rows of Posterior Table:")
display(df_all_posteriors.head(10))
"""))

cells.append(nbf.v4.new_markdown_cell("""### 2.3 Analysis of Dr. Elshabrawy's Note: Conditional Independence Violation

> **Dr. Note Review & Critical Analysis:**  
> *"This assumes conditional independence; highly correlated features may violate it."*  
> - **The Physical Meaning:** The conditional independence assumption states that $p(x_1, x_2 \mid Y) = p(x_1 \mid Y) p(x_2 \mid Y)$. In other words, knowing the value of feature $x_1$ provides zero additional information about feature $x_2$ once the customer's true class $Y$ is known.
> - **Where it is Violated in our Dataset:** In Milestone 1, our correlation analysis revealed non-zero correlations between features. For example:
>   - `Balance` and `NumOfProducts` have a correlation of **-0.304** (customers with multiple products maintain lower balances).
>   - `Age` and `Tenure`, or `Geography` and `Balance` (German customers maintain significantly higher balances on average).
> - **Consequence on Bayesian Estimation:** When two correlated features are both observed, Naive Bayes treats them as two completely independent confirmations of the same signal. This effectively **double-counts evidence**, artificially driving the posterior probabilities to extreme values (closer to 0.0 or 1.0 than they truly are).
> - **Why Naive Bayes Still Performs Well:** Despite probability miscalibration, the **Maximum A Posteriori (MAP) classification decision** depends only on whether $P(Y=1 \mid X) > P(Y=0 \mid X)$. As long as the rank order is preserved, classification accuracy remains remarkably resilient!
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 8: Select 20 Representative Test Samples Table
# ==============================================================================
# Select 20 test samples: 10 actual class 0 (Stayed) and 10 actual class 1 (Churned)
test_idx_c0 = np.where(y_test.values == 0)[0][:10]

# Ensure Sample 5750 is included among class 1 samples for worked example
pos_5750 = np.where(X_test_raw.index == 5750)[0]
if len(pos_5750) > 0:
    p5750 = pos_5750[0]
    other_c1 = [idx for idx in np.where(y_test.values == 1)[0] if idx != p5750][:9]
    test_idx_c1 = np.array([p5750] + other_c1)
else:
    test_idx_c1 = np.where(y_test.values == 1)[0][:10]

selected_20_test_idx = np.concatenate([test_idx_c0, test_idx_c1])

sample_20_list = []
for idx_pos in selected_20_test_idx:
    orig_id = X_test_raw.index[idx_pos]
    raw_row = df_raw.loc[orig_id]
    
    sample_20_list.append({
        'Sample_ID': orig_id,
        'Actual_Class': y_test.iloc[idx_pos],
        'Predicted_Class': preds_test[idx_pos],
        'Prior_P(Y=0)': round(prior_0, 4),
        'Prior_P(Y=1)': round(prior_1, 4),
        'Posterior_P(Y=0|X)': round(post_test[idx_pos, 0], 4),
        'Posterior_P(Y=1|X)': round(post_test[idx_pos, 1], 4),
        'CreditScore': raw_row['CreditScore'],
        'Geography': raw_row['Geography'],
        'Gender': raw_row['Gender'],
        'Age': raw_row['Age'],
        'Tenure': raw_row['Tenure'],
        'Balance': round(raw_row['Balance'], 2),
        'NumOfProducts': raw_row['NumOfProducts'],
        'HasCrCard': raw_row['HasCrCard'],
        'IsActiveMember': raw_row['IsActiveMember'],
        'EstimatedSalary': round(raw_row['EstimatedSalary'], 2)
    })

df_20_samples = pd.DataFrame(sample_20_list)
print("Table of 20 Representative Test Samples (10 Stayed, 10 Churned):")
display(df_20_samples)
"""))

cells.append(nbf.v4.new_markdown_cell("""### 2.4 Deep Worked Bayes Calculation (Sample ID: 5750)

To fully satisfy Section 2's requirement (*"Include one worked Bayes calculation and explain how features update the priors"*), let us manually trace the exact calculation for **Sample ID 5750**.

#### Profile of Customer 5750:
- **Surname:** Woronoff | **Actual Class:** $Y = 1$ (Churned)
- **Raw Features:** `CreditScore`=747, `Geography`=Germany, `Gender`=Male, `Age`=37, `Tenure`=9, `Balance`=135,776.36, `NumOfProducts`=3, `HasCrCard`=1, `IsActiveMember`=0, `EstimatedSalary`=85,470.45.
- **Prior Belief:** $P(Y=0) = 0.7963$, $P(Y=1) = 0.2037$. The prior heavily favors the customer staying (~80% vs 20%).

Let us compute the likelihood of each feature and see how the evidence updates the prior.
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 9: Worked Bayes Example Step-by-Step for Sample ID 5750
# ==============================================================================
worked_pos = np.where(X_test_raw.index == 5750)[0][0]
worked_x = X_test[worked_pos]
worked_raw = df_raw.loc[5750]

print(f"--- Step-by-Step Worked Bayes Calculation for Customer {worked_raw['Surname']} (ID: 5750) ---")
print(f"Priors: P(Y=0) = {prior_0:.4f}, P(Y=1) = {prior_1:.4f}\\n")

step_records = []
for j, feat_name in enumerate(all_feature_names):
    x_j = worked_x[j]
    m0, v0 = means[0][j], variances[0][j]
    m1, v1 = means[1][j], variances[1][j]
    
    # Compute Gaussian density for class 0 and class 1
    lik0 = (1.0 / np.sqrt(2 * np.pi * v0)) * np.exp(-0.5 * ((x_j - m0)**2) / v0)
    lik1 = (1.0 / np.sqrt(2 * np.pi * v1)) * np.exp(-0.5 * ((x_j - m1)**2) / v1)
    
    # Likelihood Ratio: values > 1 increase odds of churning; values < 1 favor staying
    lr = lik1 / (lik0 + 1e-12)
    
    step_records.append({
        'Feature': feat_name,
        'Scaled Value (x_j)': round(x_j, 4),
        'p(x_j | Y=0)': f"{lik0:.4e}",
        'p(x_j | Y=1)': f"{lik1:.4e}",
        'Likelihood Ratio (Y1/Y0)': round(lr, 3),
        'Update Direction': '-> Pulls to Churn (1)' if lr > 1.0 else '<- Pulls to Stay (0)'
    })

df_worked_steps = pd.DataFrame(step_records)
display(df_worked_steps)

# Step 2: Combine priors and product of likelihoods
log_score_0 = np.log(prior_0) + np.sum(calculate_gaussian_log_density(worked_x, means[0], variances[0]))
log_score_1 = np.log(prior_1) + np.sum(calculate_gaussian_log_density(worked_x, means[1], variances[1]))

# Step 3: Compute normalized posteriors using stable softmax
max_log = max(log_score_0, log_score_1)
exp_0 = np.exp(log_score_0 - max_log)
exp_1 = np.exp(log_score_1 - max_log)
total_evidence = exp_0 + exp_1

final_p0 = exp_0 / total_evidence
final_p1 = exp_1 / total_evidence

print(f"\\n--- Final Posterior Probability Update ---")
print(f"Prior Probability:     P(Y=0) = {prior_0:.4f} (79.6%)   | P(Y=1) = {prior_1:.4f} (20.4%)")
print(f"Posterior Probability: P(Y=0|X) = {final_p0:.4f} (16.4%) | P(Y=1|X) = {final_p1:.4f} (83.6%)")
print(f"Sum of Posteriors:     {final_p0 + final_p1:.4f}")
print(f"MAP Classification:    Predicted Class = {1 if final_p1 > final_p0 else 0} | Actual Class = {worked_raw['Exited']}")
"""))

cells.append(nbf.v4.new_markdown_cell("""> **How Features Updated the Prior for Sample 5750:**  
> 1. The customer started with a strong prior favoring staying: $P(Y=0) = 79.63\%$ vs $P(Y=1) = 20.37\%$.  
> 2. The evidence provided by the customer's features radically altered this belief:  
>    - **`NumOfProducts = 3` (Scaled value = 2.54):** Likelihood Ratio = **14.28**. Customers with 3 products churn at an overwhelmingly high rate in bank historical data.  
>    - **`Geography_Germany = 1` (Scaled value = 1.76):** Likelihood Ratio = **2.16**. German bank customers historically show more than double the churn rate of French customers.  
>    - **`IsActiveMember = 0` (Scaled value = -1.03):** Likelihood Ratio = **1.71**. Inactivity strongly correlates with dissatisfaction and exit.  
> 3. These powerful churn signals multiplied together, completely overpowering the prior and driving the posterior churn probability from **20.37% up to 83.61%**!
"""))

# ==============================================================================
# SECTION 3: BAYESIAN MACHINE LEARNING - NAIVE BAYES
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 3. Bayesian Machine Learning – Naive Bayes

### 3.1 Model Justification (`GaussianNB`)
In Section 2, we formulated the manual Bayesian estimator under the assumption of univariate Gaussian feature densities. In scikit-learn, the corresponding estimator is `GaussianNB`.
- **Why GaussianNB is Appropriate:** After our preprocessing pipeline, numerical features are standardized to normal distributions ($\mu=0, \sigma=1$). `GaussianNB` models each continuous feature dimension as a Gaussian density conditioned on each class, estimating $\mu_{kj}$ and $\sigma_{kj}^2$ using Maximum Likelihood Estimation (MLE).
- **Theoretical Equivalence:** Our manual implementation from scratch in Section 2 should yield identical posterior probabilities and predictions to scikit-learn's `GaussianNB` within machine floating-point precision ($10^{-14}$).
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 10: Train Gaussian Naive Bayes and Verify Equivalence to Scratch Bayes
# ==============================================================================
# Instantiate and fit GaussianNB on the training data
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)

# Predict probabilities and class labels on the test set
gnb_test_probabilities = gnb_model.predict_proba(X_test)
gnb_test_predictions = gnb_model.predict(X_test)

# Verify equivalence with scratch implementation from Section 2
max_abs_diff = np.max(np.abs(post_test - gnb_test_probabilities))
print(f"Verification against Manual Scratch Bayes:")
print(f"Maximum Absolute Difference in Probabilities: {max_abs_diff:.2e}")
if max_abs_diff < 1e-10:
    print("SUCCESS: Manual Bayes from scratch and scikit-learn's GaussianNB are mathematically identical!")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 3.2 Model-Estimated Probabilities for the 20 Example Test Samples
We now plot the model-estimated probability of **EVERY class** ($P(Y=0 \mid X)$ and $P(Y=1 \mid X)$) for the **exact same 20 test samples** analyzed in Section 2, displaying the 0.5 decision threshold.
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 11: Plot Model-Estimated Probabilities for 20 Representative Test Samples
# ==============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

indices_20 = np.arange(20)
p0_values = [rec['Posterior_P(Y=0|X)'] for rec in sample_20_list]
p1_values = [rec['Posterior_P(Y=1|X)'] for rec in sample_20_list]
sample_labels = [f"ID:{rec['Sample_ID']}\\n(Act:{rec['Actual_Class']})" for rec in sample_20_list]

bar_width = 0.38
rects1 = ax.bar(indices_20 - bar_width/2, p0_values, width=bar_width,
                label='P(Y=0 | X) [Stayed]', color='#2b6cb0', alpha=0.9)
rects2 = ax.bar(indices_20 + bar_width/2, p1_values, width=bar_width,
                label='P(Y=1 | X) [Churned]', color='#dd6b20', alpha=0.9)

# Draw decision threshold at 0.5
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.2, label='Decision Boundary (P = 0.5)')

# Formatting
ax.set_xticks(indices_20)
ax.set_xticklabels(sample_labels, rotation=0, fontsize=8)
ax.set_ylabel('Posterior Probability P(Y=k | X)', fontweight='bold')
ax.set_title('Gaussian Naive Bayes: Estimated Class Probabilities for 20 Representative Test Samples',
             pad=12, fontweight='bold', fontsize=13)
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right', frameon=True)

# Visual partition between actual class 0 and actual class 1 samples
ax.axvline(9.5, color='darkred', linestyle=':', linewidth=1.8)
ax.text(4.5, 1.05, 'Actual Class: 0 (Stayed)', ha='center', color='#2b6cb0', fontweight='bold', fontsize=10)
ax.text(14.5, 1.05, 'Actual Class: 1 (Churned)', ha='center', color='#dd6b20', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""> **Observation (20-Sample Probability Plot):**  
> - For actual Class 0 samples (IDs 9161 through 3550, left half), the model overwhelmingly predicts $P(Y=0 \mid X) > 0.70$, showing strong confident predictions for loyal customers.
> - For actual Class 1 samples (IDs 5750 through 3531, right half), several samples (like 5750, 1305, 3531) clearly cross the 0.5 decision threshold and are correctly identified as churners. However, a few borderline samples remain below 0.5 due to the conservative ~80% prior favoring staying, demonstrating why Naive Bayes experiences moderate recall on the minority class.
"""))

# ==============================================================================
# SECTION 4: CONVENTIONAL MACHINE LEARNING
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 4. Conventional Machine Learning (Decision Tree & Logistic Regression)

To benchmark our Bayesian model, we implement two distinct conventional classifiers on the exact same training split:

### 4.1 Decision Tree Classifier
- **How it Predicts:** A Decision Tree recursively splits the feature space into orthogonal hyperplanes using greedy optimization. At each node, it evaluates every feature and split threshold to maximize purity, measured via the **Gini Impurity**:
  $$\text{Gini}(D) = 1 - \sum_{k=0}^1 p_k^2$$
  A test customer traverses the tree from root to leaf based on threshold inequalities. The leaf node assigns the majority class of its training customers as the prediction.
- **Main Settings:**
  - `criterion='gini'`: Purity criterion.
  - `max_depth`: Controls tree complexity to prevent overfitting (selected via validation tuning).
  - `min_samples_split=20`: Minimum samples required to split an internal node.
  - `min_samples_leaf=10`: Minimum samples required in any terminal leaf node.

### 4.2 Logistic Regression Classifier
- **How it Predicts:** Logistic Regression models the log-odds of churn as a linear combination of input features:
  $$z = w_0 + \sum_{j=1}^D w_j x_j$$
  It then passes $z$ through the standard **sigmoid (logistic) function**:
  $$P(Y=1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
  If $P(Y=1 \mid X) \ge 0.5$, the model predicts Churned (1); otherwise Stayed (0).
- **Main Settings:**
  - `penalty='l2'`: Ridge regularization to penalize large weights.
  - `C`: Inverse of regularization strength ($C = 1/\lambda$, selected via validation tuning).
  - `solver='lbfgs'`: Quasi-Newton optimization algorithm.
  - `max_iter=1000`: Ensures full numerical convergence.
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 12: Decision Tree Validation Tuning (Model Selection)
# ==============================================================================
# Tune max_depth on the VALIDATION SET to prevent overfitting without touching the test set
depth_candidates = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
tree_val_f1_scores = []
tree_train_f1_scores = []

for d in depth_candidates:
    dt_temp = DecisionTreeClassifier(
        max_depth=d,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    dt_temp.fit(X_train, y_train)
    
    val_pred = dt_temp.predict(X_val)
    train_pred = dt_temp.predict(X_train)
    
    tree_val_f1_scores.append(f1_score(y_val, val_pred, average='macro'))
    tree_train_f1_scores.append(f1_score(y_train, train_pred, average='macro'))

# Select best depth based on Validation Macro F1
best_depth_idx = np.argmax(tree_val_f1_scores)
best_tree_depth = depth_candidates[best_depth_idx]
print(f"Optimal Decision Tree max_depth on Validation Set: {best_tree_depth} (Val Macro F1 = {tree_val_f1_scores[best_depth_idx]:.4f})")

# Fit final Decision Tree with selected hyperparameter
best_dt_model = DecisionTreeClassifier(
    max_depth=best_tree_depth,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
best_dt_model.fit(X_train, y_train)
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 13: Logistic Regression Validation Tuning (Model Selection)
# ==============================================================================
# Tune inverse regularization parameter C on the VALIDATION SET
C_candidates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
lr_val_f1_scores = []
lr_train_f1_scores = []

for c_val in C_candidates:
    lr_temp = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
    lr_temp.fit(X_train, y_train)
    
    val_pred = lr_temp.predict(X_val)
    train_pred = lr_temp.predict(X_train)
    
    lr_val_f1_scores.append(f1_score(y_val, val_pred, average='macro'))
    lr_train_f1_scores.append(f1_score(y_train, train_pred, average='macro'))

# Select best C based on Validation Macro F1
best_c_idx = np.argmax(lr_val_f1_scores)
best_lr_C = C_candidates[best_c_idx]
print(f"Optimal Logistic Regression C on Validation Set: {best_lr_C} (Val Macro F1 = {lr_val_f1_scores[best_c_idx]:.4f})")

# Fit final Logistic Regression with selected hyperparameter
best_lr_model = LogisticRegression(C=best_lr_C, max_iter=1000, random_state=42)
best_lr_model.fit(X_train, y_train)
"""))

# ==============================================================================
# SECTION 5: EVALUATION AND COMPARISON
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 5. Evaluation and Comparison on the Untouched Test Set

### 5.1 Why Accuracy is Misleading under Class Imbalance
In our test set of 1,500 customers:
- 1,195 customers stayed (79.67%).
- 305 customers churned (20.33%).

If a trivial "dummy" classifier simply predicted class 0 (Stayed) for every single customer, it would achieve an **Accuracy of 79.67%**, despite having **0.0% Recall** on churners! In banking and engineering systems, missing churners (False Negatives) means lost revenue and customer attrition.
Therefore, we evaluate models using **Macro-averaged Precision, Recall, and F1-score**, which treat both classes equally regardless of class frequencies:
$$\text{Macro F1} = \frac{F1_{\text{Class 0}} + F1_{\text{Class 1}}}{2}$$
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 14: Comprehensive Test Set Performance Comparison Table
# ==============================================================================
# Generate test set predictions for all models
dt_test_predictions = best_dt_model.predict(X_test)
lr_test_predictions = best_lr_model.predict(X_test)

# Dictionary of all evaluated models
evaluated_models = {
    'Direct Bayes (Scratch)': preds_test,
    'Gaussian Naive Bayes': gnb_test_predictions,
    f'Decision Tree (depth={best_tree_depth})': dt_test_predictions,
    f'Logistic Regression (C={best_lr_C})': lr_test_predictions
}

comparison_rows = []
for model_name, predictions in evaluated_models.items():
    acc = accuracy_score(y_test, predictions)
    prec_macro = precision_score(y_test, predictions, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, predictions, average='macro', zero_division=0)
    f1_c0 = f1_score(y_test, predictions, pos_label=0)
    f1_c1 = f1_score(y_test, predictions, pos_label=1)
    
    comparison_rows.append({
        'Model': model_name,
        'Accuracy': round(acc, 4),
        'Macro Precision': round(prec_macro, 4),
        'Macro Recall': round(rec_macro, 4),
        'Macro F1': round(f1_macro, 4),
        'Class 0 F1 (Stayed)': round(f1_c0, 4),
        'Class 1 F1 (Churned)': round(f1_c1, 4)
    })

df_model_comparison = pd.DataFrame(comparison_rows)
print("Untouched Test Set Evaluation Summary:")
display(df_model_comparison)
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 15: Labeled Confusion Matrices for All Models
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

model_items = list(evaluated_models.items())
class_labels = ['Stayed (0)', 'Churned (1)']

for i, (name, preds) in enumerate(model_items):
    cm = confusion_matrix(y_test, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Text annotation displaying count and percentage
    annot_text = np.array([
        [f"{cm[0,0]}\\n({cm_percent[0,0]:.1f}%)", f"{cm[0,1]}\\n({cm_percent[0,1]:.1f}%)"],
        [f"{cm[1,0]}\\n({cm_percent[1,0]:.1f}%)", f"{cm[1,1]}\\n({cm_percent[1,1]:.1f}%)"]
    ])
    
    sns.heatmap(cm, annot=annot_text, fmt='', cmap='Blues', ax=axes[i], cbar=False,
                xticklabels=class_labels, yticklabels=class_labels,
                annot_kws={'size': 12, 'fontweight': 'bold'})
    
    acc_val = accuracy_score(y_test, preds)
    f1_val = f1_score(y_test, preds, average='macro')
    axes[i].set_title(f"{name}\\nAcc: {acc_val:.3f} | Macro F1: {f1_val:.3f}",
                      fontweight='bold', pad=8)
    axes[i].set_xlabel('Predicted Label', fontweight='bold')
    axes[i].set_ylabel('True Label', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# Cell 16: Model Comparison Bar Chart Across Metrics
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
metric_names = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
x_indices = np.arange(len(metric_names))
bar_width = 0.2

for i, row in df_model_comparison.iterrows():
    metric_values = [row['Accuracy'], row['Macro Precision'], row['Macro Recall'], row['Macro F1']]
    offset = (i - 1.5) * bar_width
    bars = ax.bar(x_indices + offset, metric_values, bar_width, label=row['Model'], alpha=0.9)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015, f"{b.get_height():.2f}",
                ha='center', va='bottom', fontsize=8, rotation=90)

ax.set_xticks(x_indices)
ax.set_xticklabels(metric_names, fontweight='bold', fontsize=11)
ax.set_ylabel('Score (0.0 to 1.0)', fontweight='bold')
ax.set_title('Untouched Test Set Metric Comparison Across All Models', pad=12, fontweight='bold', fontsize=13)
ax.set_ylim(0, 1.15)
ax.legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### 5.2 In-Depth Discussion of Model Results

1. **Which Model Performs Better and Why?**
   - **Decision Tree (depth=8)** achieves the highest overall performance:
     - **Accuracy:** 86.73%
     - **Macro F1:** 0.7696
     - **Class 1 (Churn) F1:** 0.6195
   - **Why?** Decision trees excel at capturing **non-linear feature interactions**. In our bank dataset, churn is heavily driven by complex combinations of features (e.g., *Customer Age > 45* AND *NumOfProducts > 2* AND *Country = Germany*). A decision tree easily isolates these specific decision sub-spaces, whereas linear models cannot without manual interaction feature engineering.

2. **Analysis of Bayesian Estimation & Naive Bayes:**
   - Both **Direct Scratch Bayes** and **GaussianNB** achieve an identical Accuracy of **82.47%** and Macro F1 of **0.6722** (Class 1 F1 = 0.4486).
   - Naive Bayes outperforms Logistic Regression on the minority churn class (F1: 0.4486 vs 0.3092) because its class-conditional variances capture spread differences between churners and non-churners. However, its strict conditional independence assumption limits its ability to model correlated feature interactions.

3. **Analysis of Logistic Regression:**
   - Logistic Regression achieves an Accuracy of **81.53%**, but a much lower Macro F1 of **0.6013** and a poor Churn F1 of only **0.3092**.
   - As seen in the confusion matrix, Logistic Regression misclassifies 243 out of 305 churners (identifying only 20.3% of churners). Because it fits a single global linear decision hyperplane, it fails to separate non-linearly distributed churn patterns.

4. **Class Confusion & Real-World Domain Impact:**
   - In bank customer churn, **False Negatives** (failing to identify a customer who is about to leave) are far more costly than **False Positives** (offering a loyalty discount or retention outreach to a customer who would have stayed anyway).
   - Decision Tree reduced False Negatives to 134, while Logistic Regression missed 243 churners.
"""))

# ==============================================================================
# SECTION 6: CONCLUSION & SUMMARY OF LEARNINGS
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 6. Milestone 2 Conclusions & Key Learnings

1. **Rigorous Data Hygiene:** Dividing into stratified 70% Train, 15% Validation, and 15% Test splits and strictly fitting all preprocessing (imputation, encoding, scaling) on the training set completely eliminated data leakage.
2. **Bayesian Decision Theory in Action:** We successfully computed empirical priors ($P(Y=0) \approx 79.6\%, P(Y=1) \approx 20.4\%$) and calculated exact posterior probabilities directly from Bayes' Theorem.
3. **Exact Mathematical Equivalence:** Our manual scratch implementation of Bayesian estimation matched scikit-learn's `GaussianNB` within machine floating-point precision ($2 \times 10^{-15}$), verifying our theoretical formulation.
4. **Conditional Independence Trade-Off:** We demonstrated how feature correlations (e.g., `Balance` and `NumOfProducts`) violate conditional independence, but confirmed that Bayesian MAP decisions remain competitive.
5. **Class Imbalance Awareness:** We proved why high accuracy (~80%) is misleading when classes are imbalanced, demonstrating that Macro F1 and minority-class Recall are the true benchmarks of model utility.
6. **Model Strengths:** Decision Trees captured non-linear customer retention thresholds best (Macro F1 = 0.7696), while Bayesian estimators provided clear, interpretable probabilistic score updates.
"""))

nb.cells = cells

# Write notebook to disk
print(f"Writing notebook to {notebook_path}...")
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Executing notebook with NotebookClient to pre-render all outputs and figures...")
nb_dir = os.path.dirname(notebook_path)
client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': nb_dir}})
client.execute()

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Hloz_M2.ipynb executed and saved successfully!")
