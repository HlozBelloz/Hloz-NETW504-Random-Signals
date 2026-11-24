# Milestone 2: Comprehensive Code Walkthrough, Mathematical Derivations & Evaluation Defense Guide

**Course:** Random Signals and Noise (NETW504)  
**Instructor:** Prof. Talal Elshabrawy  
**Team Name:** Hloz  
**Team Members:**  
- Mazen Mohamed Hamdy Altelbany (ID: 64-12371)  
- Malak Sherif Mohamed (ID: 64-10784)  
- Suhad Eyhab Rasheed (ID: 64-31506)  
- Sama Ismael Ahel (ID: 64-19880)  

---

## 📖 Table of Contents
1. [Executive Summary & Milestone 2 Scope](#1-executive-summary--milestone-2-scope)
2. [Mathematical Foundations & Formulas](#2-mathematical-foundations--formulas)
3. [The Preprocessing Pipeline & Data Leakage Prevention](#3-the-preprocessing-pipeline--data-leakage-prevention)
4. [Cell-by-Cell Notebook Code Walkthrough](#4-cell-by-cell-notebook-code-walkthrough)
5. [In-Depth Worked Bayes Calculation (Sample 5750)](#5-in-depth-worked-bayes-calculation-sample-5750)
6. [Conventional Machine Learning Mechanics (Decision Tree & Logistic Regression)](#6-conventional-machine-learning-mechanics)
7. [Evaluation Results, Confusion Matrices & Imbalance Discussion](#7-evaluation-results-confusion-matrices--imbalance-discussion)
8. [Professor Defense Preparation: Expected Questions & Defensible Answers](#8-professor-defense-preparation-expected-questions--defensible-answers)

---

## 1. Executive Summary & Milestone 2 Scope

The goal of **Milestone 2** is to move from descriptive statistics (Milestone 1) to **Bayesian Decision Theory**, statistical likelihood estimation, and supervised machine learning classification.

### What was completed:
1. **Dataset Partitioning:** Partitioned into stratified **70% Training (7,000 samples)**, **15% Validation (1,500 samples)**, and **15% Test (1,500 samples)**.
2. **Data Leakage Prevention:** Built a preprocessing pipeline (median imputation, mode imputation, one-hot encoding with `drop='first'`, and `StandardScaler`) fitted **strictly on the Training set only**, then applied to transform Validation and Test.
3. **Manual Bayesian Estimator from Scratch:**
   - Computed empirical training class priors: $P(Y=0) = 0.7963$, $P(Y=1) = 0.2037$.
   - Modeled feature likelihoods using Gaussian density functions $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$ with variance smoothing $\epsilon = 10^{-9}\sigma_{\max}^2$.
   - Evaluated Bayes' Theorem in log-space with stable softmax (log-sum-exp trick), ensuring all class posterior probabilities sum to $1.0$.
   - Saved all 10,000 posteriors into `Hloz_M2_Bayesian_Posteriors.csv`.
4. **Detailed 20-Sample Study & Worked Example:**
   - Selected 20 test samples (10 stayed, 10 churned).
   - Produced a step-by-step manual calculation for **Sample ID 5750** showing how feature evidence multiplied together to increase the churn odds from a prior of $20.37\%$ to a posterior of $83.61\%$.
5. **Bayesian Machine Learning (`GaussianNB`):**
   - Trained scikit-learn's `GaussianNB` and verified that its probability outputs are **identical to our manual scratch Bayes implementation** (maximum absolute difference = $2.00 \times 10^{-15}$).
   - Plotted class probabilities for the 20 test samples showing the $0.5$ decision boundary.
6. **Conventional Machine Learning:**
   - Trained a **Decision Tree** and tuned `max_depth` on the Validation Set (optimal depth = 8).
   - Trained **Logistic Regression** and tuned inverse regularization $C$ on the Validation Set (optimal $C = 1.0$).
7. **Model Comparison on Untouched Test Set:**
   - Benchmarked all models on Accuracy, Macro Precision, Macro Recall, Macro F1, and Class F1 scores.
   - Plotted labeled confusion matrices with counts and class recall percentages.
   - Documented the business and statistical reasons why Decision Tree won (Macro F1 = 0.7696) and why accuracy is deceptive under class imbalance.
8. **Deliverables Created:**
   - `Hloz_M2.ipynb` (fully executed with outputs, tables, and plots).
   - `Hloz_M2_Report.pdf` (strictly 6 pages, professional layout).
   - `Hloz_M2_Bayesian_Posteriors.csv` (10,000-row posterior table).

---

## 2. Mathematical Foundations & Formulas

### 2.1 Bayes' Theorem for Classification
For a customer with feature vector $X = [x_1, x_2, \dots, x_D]^T$ and class label $Y \in \{0, 1\}$:

$$P(Y=k \mid X=x) = \frac{P(Y=k) \cdot p(X=x \mid Y=k)}{p(X=x)} = \frac{P(Y=k) \cdot p(X=x \mid Y=k)}{\sum_{c \in \{0, 1\}} P(Y=c) \cdot p(X=x \mid Y=c)}$$

- **Prior $P(Y=k)$:** The baseline probability of a class based on historical training counts:
  $$P(Y=k) = \frac{N_{\text{train}, k}}{N_{\text{train}}}$$
- **Conditional Independence Assumption (Naive Bayes):**
  $$p(X=x \mid Y=k) = \prod_{j=1}^D p(x_j \mid Y=k)$$
- **Gaussian Likelihood Density:** For continuous standardized feature $x_j$:
  $$p(x_j \mid Y=k) = \frac{1}{\sqrt{2\pi \sigma_{kj}^2}} \exp\left(-\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right)$$
  Where $\mu_{kj}$ and $\sigma_{kj}^2$ are the training sample mean and variance of feature $j$ in class $k$.

### 2.2 Numerical Underflow & The Log-Sum-Exp Trick
Multiplying $D=11$ probability densities (each often around $0.01$ to $0.5$) yields values on the order of $10^{-15}$ to $10^{-30}$. In floating-point arithmetic, this risks precision loss or complete underflow to $0.0$.
To prevent this, we calculate in log-space:

$$\log \text{score}_k(x) = \log P(Y=k) + \sum_{j=1}^D \log p(x_j \mid Y=k)$$

Where the Gaussian log-density is:
$$\log p(x_j \mid Y=k) = -\frac{1}{2}\log(2\pi \sigma_{kj}^2) - \frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}$$

To normalize back to probabilities without numerical overflow or underflow:
Let $M = \max(\log \text{score}_0, \log \text{score}_1)$.
$$P(Y=k \mid x) = \frac{\exp(\log \text{score}_k - M)}{\sum_{c} \exp(\log \text{score}_c - M)}$$
This is the **stable softmax (log-sum-exp)** technique used in production deep learning and statistical signal processing.

### 2.3 Likelihood Ratio Test (LRT) & Prior Updating
The posterior odds ratio relates directly to the prior odds ratio via the Likelihood Ratio ($LR$):

$$\frac{P(Y=1 \mid X)}{P(Y=0 \mid X)} = \underbrace{\frac{P(Y=1)}{P(Y=0)}}_{\text{Prior Odds}} \times \underbrace{\prod_{j=1}^D \frac{p(x_j \mid Y=1)}{p(x_j \mid Y=0)}}_{\text{Likelihood Ratio Product (Evidence)}}$$

- **If $LR_j > 1$:** Feature $j$ provides evidence that increases the odds of churning.
- **If $LR_j < 1$:** Feature $j$ provides evidence that the customer will stay.
- **Decision Boundary:** The Maximum A Posteriori (MAP) rule chooses class 1 if posterior odds $> 1$, meaning $P(Y=1 \mid X) > 0.5$.

---

## 3. The Preprocessing Pipeline & Data Leakage Prevention

### What is Data Leakage?
Data leakage occurs when information from outside the training dataset is used to train or scale the model.
- **Common Rookie Mistake:** Running `StandardScaler().fit_transform(df)` on the whole 10,000 rows *before* splitting.
  - *Why this is wrong:* The scaler computes the mean $\mu$ and standard deviation $\sigma$ across all 10,000 samples. When you later test on the test set, your model has already "seen" the test set's distribution parameters!
- **Our Strict Clean Architecture:**
  1. Split raw data into Train (70%), Val (15%), Test (15%) using stratification.
  2. Instantiate `ColumnTransformer` with `SimpleImputer` and `StandardScaler` for numeric, and `OneHotEncoder(drop='first')` for categorical.
  3. Call `.fit(X_train_raw)` on the **Training set ONLY**.
  4. Call `.transform()` on Train, Validation, and Test separately.

---

## 4. Cell-by-Cell Notebook Code Walkthrough

Here is a detailed line-by-line explanation of every cell in `Hloz_M2.ipynb`:

### Cell 1: Environment Setup, Library Imports & Raw Data Loading
```python
# We import NumPy for fast vectorized math, Pandas for DataFrames,
# Matplotlib and Seaborn for plotting, and scikit-learn modules for splitting and modeling.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load raw CSV and drop ID columns ('RowNumber', 'CustomerId', 'Surname')
# because these are arbitrary database identifiers with no causal relationship to churn.
df_raw = pd.read_csv('../data set/Churn_Modelling.csv')
id_columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
df = df_raw.drop(columns=id_columns_to_drop)
```
*Why this matters:* Dropping ID columns prevents models from memorizing spurious customer keys, leaving 10 predictive features.

### Cell 2: Two-Step Stratified Dataset Splitting
```python
# Step 1: Hold out 30% for validation and testing combined
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
# Step 2: Split the 30% holdout exactly in half -> 15% validation, 15% test
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
```
*Why two steps?* `train_test_split` only splits into two sets at a time. By splitting 70/30 first and then 50/50 on the temporary set, we obtain an exact 70% (7,000), 15% (1,500), and 15% (1,500) split. `stratify=y` guarantees that every subset maintains exactly 20.37% churners.

### Cell 3: Constructing & Fitting Preprocessor on Training Data ONLY
```python
# Define numerical pipeline: median imputation + standardization
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define categorical pipeline: mode imputation + one-hot encoding (drop='first' to avoid dummy trap)
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# FIT ON TRAINING ONLY
preprocessor.fit(X_train_raw)

# Transform splits
X_train = preprocessor.transform(X_train_raw)
X_val = preprocessor.transform(X_val_raw)
X_test = preprocessor.transform(X_test_raw)
```
*Why `drop='first'`?* `Geography` has 3 categories (`France`, `Germany`, `Spain`). If we kept all 3, `France + Germany + Spain = 1`, creating perfect multicollinearity. Dropping `France` creates 2 independent binary columns, where France is the reference baseline $(0, 0)$.

### Cell 4: Computing Empirical Training Priors
```python
N_train = len(y_train) # 7,000
N_class_0 = (y_train == 0).sum() # 5,574
N_class_1 = (y_train == 1).sum() # 1,426

prior_0 = N_class_0 / N_train # 0.7963
prior_1 = N_class_1 / N_train # 0.2037
```
*Insight:* Class 0 is ~3.91 times larger than Class 1. Any random customer has an 80% baseline probability of staying before we even look at their features.

### Cell 5: Estimating Feature Means and Variances per Class
```python
# For each class, compute mean vector and variance vector across all 11 features
var_smoothing = 1e-9 * np.max(np.var(X_train, axis=0)) # epsilon to prevent division by zero
for k in [0, 1]:
    X_k = X_train[y_train.values == k]
    means[k] = np.mean(X_k, axis=0)
    variances[k] = np.var(X_k, axis=0) + var_smoothing
```
*Why variance smoothing?* If a feature has zero variance in a subset, dividing by $\sigma^2$ causes division by zero. Adding $\epsilon = 10^{-9}\sigma_{\max}^2$ matches scikit-learn's exact numerical stability formula.

### Cell 6: Direct Bayesian Posterior Calculation (From Scratch)
```python
def calculate_gaussian_log_density(x, mu, var):
    # Evaluates log of 1/sqrt(2*pi*var) * exp(-(x - mu)^2 / (2*var))
    return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * ((x - mu) ** 2) / var

def compute_bayesian_posteriors(X_matrix):
    # Calculates log score_k = log P(Y=k) + sum_j log p(x_ij | Y=k)
    # Applies stable softmax to normalize
    ...
```
*Verification:* An explicit assertion `assert np.allclose(np.sum(post_test, axis=1), 1.0)` confirms that every row's probabilities sum to $1.0$.

### Cell 7: Exporting Full 10,000-Row Posteriors Table
Saves all samples with `Sample_ID`, `Split`, `Actual_Class`, `P(Y=0|X)`, `P(Y=1|X)`, and `Predicted_Class` to `Hloz_M2_Bayesian_Posteriors.csv`.

### Cell 8: 20 Representative Test Samples Table
Extracts 10 actual stayed and 10 actual churned test samples, displaying their feature values, priors, posteriors, and predicted classes.

### Cell 9: Worked Step-by-Step Bayes Calculation for Sample 5750
Breaks down feature likelihoods, likelihood ratios, unnormalized log scores, and final normalized posteriors ($P(Y=0|X) = 0.1639, P(Y=1|X) = 0.8361$).

### Cell 10: Training `GaussianNB` & Equivalence Check
Fits `GaussianNB()` on `X_train` and computes `predict_proba(X_test)`.
Checks: `max_abs_diff = np.max(np.abs(post_test - gnb_test_probabilities))`.
Result: $2.00 \times 10^{-15}$. This confirms our manual code is mathematically exact!

### Cell 11: 20-Sample Probability Bar Chart
Plots $P(Y=0 \mid X)$ and $P(Y=1 \mid X)$ side-by-side for each sample with the 0.5 decision threshold line.

### Cells 12 & 13: Decision Tree & Logistic Regression Validation Tuning
Evaluates tree depths $d \in [2, 15]$ and regularization strengths $C \in [0.001, 100.0]$ on the **Validation Set**.
Selects optimal hyperparameters ($d = 8$, $C = 1.0$) based on Validation Macro F1.

### Cells 14, 15 & 16: Untouched Test Set Evaluation
Generates the comparison table, labeled confusion matrices, and multi-metric bar chart for all models on the untouched 1,500 test samples.

---

## 5. In-Depth Worked Bayes Calculation (Sample 5750)

### Profile of Customer 5750 (Woronoff):
- **Actual Class:** Churned ($Y = 1$)
- **Prior Belief:** $P(Y=0) = 0.7963$, $P(Y=1) = 0.2037$ (Prior Odds = $0.2037 / 0.7963 \approx 0.256$)

### Step-by-Step Feature Evidence Breakdown:

| Feature Name | Raw Value | Scaled Value ($x_j$) | Likelihood $p(x_j \mid Y=0)$ | Likelihood $p(x_j \mid Y=1)$ | Likelihood Ratio ($LR_j$) | Bayesian Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CreditScore` | 747 | 1.0003 | $2.468 \times 10^{-1}$ | $2.252 \times 10^{-1}$ | 0.912 | Weak pull to Stay |
| `Age` | 37 | -0.1873 | $4.094 \times 10^{-1}$ | $3.195 \times 10^{-1}$ | 0.780 | Weak pull to Stay |
| `Tenure` | 9 | 1.3807 | $1.525 \times 10^{-1}$ | $1.596 \times 10^{-1}$ | 1.046 | Neutral / Slight Churn |
| `Balance` | $135,776.36 | 0.9526 | $2.393 \times 10^{-1}$ | $3.153 \times 10^{-1}$ | 1.318 | Moderate pull to Churn |
| **`NumOfProducts`** | **3** | **2.5147** | **$7.798 \times 10^{-3}$** | **$5.067 \times 10^{-2}$** | **6.498** | **Extremely Strong Churn** |
| `HasCrCard` | 1 | 0.6441 | $3.254 \times 10^{-1}$ | $3.205 \times 10^{-1}$ | 0.985 | Neutral |
| **`IsActiveMember`** | **0** | **-1.0264** | **$2.151 \times 10^{-1}$** | **$3.085 \times 10^{-1}$** | **1.434** | **Moderate pull to Churn** |
| `EstimatedSalary` | $85,470.45 | -0.2443 | $3.896 \times 10^{-1}$ | $3.786 \times 10^{-1}$ | 0.972 | Neutral |
| **`Geography_Germany`** | **1** | **1.7614** | **$1.515 \times 10^{-1}$** | **$3.880 \times 10^{-1}$** | **2.561** | **Strong Churn Signal** |
| `Geography_Spain` | 0 | -0.5735 | $7.581 \times 10^{-1}$ | $8.708 \times 10^{-1}$ | 1.149 | Slight Churn |
| `Gender_Male` | 1 | 0.9087 | $5.613 \times 10^{-1}$ | $4.347 \times 10^{-1}$ | 0.774 | Weak pull to Stay |

### Score Aggregation:
$$\log \text{score}_0 = \log(0.7963) + \sum_{j=1}^{11} \log p(x_j \mid Y=0) = -0.2278 - 19.2603 = -19.4881$$
$$\log \text{score}_1 = \log(0.2037) + \sum_{j=1}^{11} \log p(x_j \mid Y=1) = -1.5911 - 16.2665 = -17.8576$$

Notice that $\log \text{score}_1$ is significantly larger (less negative) than $\log \text{score}_0$ by a margin of $\Delta = 1.6305$!

### Normalization (Stable Softmax):
$$M = \max(-19.4881, -17.8576) = -17.8576$$
$$\exp(\log \text{score}_0 - M) = \exp(-1.6305) = 0.1958$$
$$\exp(\log \text{score}_1 - M) = \exp(0.0) = 1.0000$$
$$\text{Total Evidence} = 0.1958 + 1.0000 = 1.1958$$

$$P(Y=0 \mid X) = \frac{0.1958}{1.1958} = \mathbf{0.1639 \ (16.39\%)}$$
$$P(Y=1 \mid X) = \frac{1.0000}{1.1958} = \mathbf{0.8361 \ (83.61\%)}$$

$$\text{MAP Decision}: \hat{Y} = \arg\max_k P(Y=k \mid X) = \mathbf{1 \ (Churned)}$$
The prediction matches ground truth!

---

## 6. Conventional Machine Learning Mechanics

### 6.1 Decision Tree Classifier
1. **Splitting Rule:** At every candidate split $s$ on feature $j$, it splits data into left ($D_L$) and right ($D_R$) child nodes. It evaluates the weighted Gini impurity reduction:
   $$\Delta \text{Gini} = \text{Gini}(D) - \left(\frac{|D_L|}{|D|} \text{Gini}(D_L) + \frac{|D_R|}{|D|} \text{Gini}(D_R)\right)$$
2. **Prediction:** A test customer follows decision rules down to a terminal leaf. The leaf outputs $\hat{y} = \text{mode}(y_{\text{leaf}})$.
3. **Hyperparameter Selection:** On our Validation Set, `max_depth = 8` provided the peak Macro F1 ($0.7417$). Depths $< 5$ underfit; depths $> 10$ overfit to training noise.

### 6.2 Logistic Regression Classifier
1. **Linear Formulation:** Computes log-odds score:
   $$z = w_0 + w_1 x_1 + \dots + w_{11} x_{11}$$
2. **Sigmoid Activation:**
   $$P(Y=1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
3. **Optimization:** Minimizes cross-entropy loss with L2 regularization penalty:
   $$\mathcal{L}(w) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \sigma(z_i) + (1 - y_i) \log(1 - \sigma(z_i)) \right] + \frac{1}{2C} \|w\|_2^2$$
4. **Hyperparameter Selection:** On our Validation Set, $C = 1.0$ provided the optimal balance between regularization and sensitivity.

---

## 7. Evaluation Results, Confusion Matrices & Imbalance Discussion

### 7.1 Confusion Matrix Comparison (Test Set: 1,500 Samples)

#### 1. Direct Scratch Bayes & GaussianNB:
```
               Predicted Stayed (0)    Predicted Churned (1)
True Stayed (0)      1130 (94.6%)             65 (5.4%)
True Churned (1)      198 (64.9%)            107 (35.1%)
```
- True Negatives: 1,130 | False Positives: 65
- False Negatives: 198 | True Positives: 107
- Churn Recall: **35.1%** | Precision on Churn: **62.2%**

#### 2. Decision Tree (`max_depth = 8`):
```
               Predicted Stayed (0)    Predicted Churned (1)
True Stayed (0)      1139 (95.3%)             56 (4.7%)
True Churned (1)      143 (46.9%)            162 (53.1%)
```
- True Negatives: 1,139 | False Positives: 56
- False Negatives: 143 | True Positives: 162
- Churn Recall: **53.1%** | Precision on Churn: **74.3%**

#### 3. Logistic Regression ($C = 1.0$):
```
               Predicted Stayed (0)    Predicted Churned (1)
True Stayed (0)      1161 (97.2%)             34 (2.8%)
True Churned (1)      243 (79.7%)             62 (20.3%)
```
- True Negatives: 1,161 | False Positives: 34
- False Negatives: 243 | True Positives: 62
- Churn Recall: **20.3%** | Precision on Churn: **64.6%**

### 7.2 Why Accuracy is Deceptive
Notice that Logistic Regression has an Accuracy of **81.53%**, which seems acceptable at first glance.
However, looking at the confusion matrix reveals that Logistic Regression **missed 243 out of 305 churners (79.7% False Negative rate!)**.
Because 79.67% of customers in the test set stayed, a model that simply predicts 0 every time gets 79.67% accuracy while accomplishing zero churn detection.
**Conclusion:** In imbalanced classification, **Macro F1** and **Minority Recall** are the only honest metrics.

---

## 8. Professor Defense Preparation: Expected Questions & Defensible Answers

### Q1: "Why did you fit the scalers and encoders on the training set only, rather than the entire dataset before splitting?"
> **Answer:**  
> "Fitting on the entire dataset is a serious methodological error called **data leakage**. Standard scaling computes the mean $\mu$ and variance $\sigma^2$ across all data. If computed before splitting, the test set's distribution parameters leak into the training features, biasing the learned parameters and giving artificially optimistic performance. By strictly fitting only on $X_{\text{train}}$ and using `.transform()` on validation and test, our validation and test sets remain truly unseen out-of-sample data."

### Q2: "What happens if you balance the training data using SMOTE or undersampling?"
> **Answer:**  
> "Balancing the training data artificially changes the empirical prior from $P(Y=0) \approx 0.80, P(Y=1) \approx 0.20$ to $P(Y=0) = P(Y=1) = 0.50$. In Bayes' theorem, removing the penalty against the minority class shifts the decision boundary, boosting sensitivity (recall) for churners. However, in the real banking population, the true base rate is still 20%. An unadjusted 50/50 model deployed in production will generate an unacceptably high rate of False Positives. To fix this, model probabilities must be re-calibrated using Bayes' rule by multiplying by the true prior odds ratio."

### Q3: "Dr. Elshabrawy left a note: 'This assumes conditional independence; highly correlated features may violate it.' What did you find?"
> **Answer:**  
> "In Milestone 1, our correlation heatmap revealed non-zero correlations, notably between `Balance` and `NumOfProducts` (-0.304), and between German residence and higher balances. When features are correlated, Naive Bayes treats them as two completely separate independent confirmations, effectively double-counting the evidence. This pushes posterior probabilities towards extreme overconfidence (closer to 0.0 or 1.0). However, the classification decision rule relies only on the argmax ranking, which is why Naive Bayes still achieved an 82.47% accuracy and remained competitive."

### Q4: "Why did Decision Tree outperform Logistic Regression and Naive Bayes?"
> **Answer:**  
> "Customer churn in banking is governed by **non-linear feature interactions**. For example, having 3 products alone is risky, but having 3 products *combined* with being inactive and residing in Germany creates an extreme churn likelihood. A Decision Tree creates rectangular partitions that easily isolate these complex multi-feature regions. Logistic Regression fits a single linear hyperplane, which cannot separate non-linear combinations, while Naive Bayes assumes features don't interact conditionally."

### Q5: "How did you prove that your scratch Bayesian implementation was correct?"
> **Answer:**  
> "We compared our scratch posterior probabilities against scikit-learn's `GaussianNB.predict_proba()` across all 1,500 test samples. The maximum absolute difference between our manual calculations and scikit-learn was $2.00 \times 10^{-15}$, which is within machine floating-point precision ($10^{-16}$). This mathematically proves that our manual implementation of Bayes' Theorem, Gaussian likelihoods, and variance smoothing is identical to scikit-learn's underlying algorithm."

