# Milestone 1: Comprehensive Project Walkthrough & Explanation

**Course:** Random Signals and Noise (NETW504)  
**Instructor:** Prof. Talal Elshabrawy  
**Team Name:** Hloz  
**Team Members:**  
- Mazen Mohamed Hamdy Altelbany (ID: 64-12371)  
- Malak Sherif Mohamed (ID: 64-10784)  
- Suhad Eyhab Rasheed (ID: 64-31506)  
- Sama Ismael Ahel (ID: 64-19880)  
**GitHub Repository:** [https://github.com/HlozBelloz/Hloz-NETW504-Random-Signals](https://github.com/HlozBelloz/Hloz-NETW504-Random-Signals)  

---

## 📖 Table of Contents
1. [Overview & Project Objective](#1-overview--project-objective)
2. [What Was Done (Step-by-Step)](#2-what-was-done-step-by-step)
3. [The 3 Official Submission Deliverables](#3-the-3-official-submission-deliverables)
4. [Dataset & Statistical Findings](#4-dataset--statistical-findings)
5. [The Basic Preprocessing Pipeline](#5-the-basic-preprocessing-pipeline)
6. [How Everything Works & How to Run it Manually](#6-how-everything-works--how-to-run-it-manually)
7. [Repository File Map](#7-repository-file-map)

---

## 1. Overview & Project Objective

The goal of Milestone 1 is to perform **Exploratory Data Analysis (EDA)** and **Basic Preprocessing** on an approved classification dataset with $\ge 10,000$ samples and 2 to 4 target classes, using Python and Jupyter Notebook.

- **Chosen Dataset:** Bank Customer Churn Prediction (`Churn_Modelling.csv`) from Kaggle.
- **Samples:** 10,000 rows.
- **Target Variable:** `Exited` (Binary: `0` = Customer stayed, `1` = Customer left).

---

## 2. What Was Done (Step-by-Step)

Here is a summary of all actions performed in this milestone:

### Step 1: Environment & Dependency Setup
- Configured Python 3.12 environment.
- Installed required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `jupyter`, `reportlab`, and `pymupdf`.
- Copied relevant helper skills (`pdf`, `writing-guidelines`, `unit-test`, etc.) into `.agents/skills/`.

### Step 2: Data Quality & Sanity Check
- Examined dataset dimensions: **10,000 rows × 14 columns**.
- Checked for exact duplicate rows: **0 duplicates**.
- Checked for missing values across all columns: **0 missing values (100% complete)**.
- Analyzed class balance:
  - Class `0` (Stayed): 7,963 customers (79.63%)
  - Class `1` (Churned): 2,037 customers (20.37%)
- Identified and marked 3 ID-like columns for removal: `RowNumber`, `CustomerId`, `Surname`.

### Step 3: Statistical Computation
- For all 8 numerical input features (`CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`), computed:
  - `count`, `mean`, `standard deviation`, `min`, `median`, and `max`.

### Step 4: Generation of Mandatory EDA Plots
Generated high-resolution plots with clear titles and labeled axes:
1. **Target Distribution Bar Plot:** Visualizes the ~80/20 class imbalance.
2. **Missing Values Bar Plot:** Confirms 0 nulls across all 14 columns.
3. **Estimated PDF for All 8 Numerical Features:** Normalized histograms (`density=True`) with Gaussian KDE curves.
4. **Empirical CDF for All 8 Numerical Features:** Step plots of cumulative probability $F_X(x)$.
5. **Correlation Heatmap:** Post-cleaning Pearson correlation matrix.
6. **Key Feature Boxplots:** Comparative distribution of `Age` and `CreditScore` by churn status.

### Step 5: Strict 4-Step Basic Preprocessing
Applied preprocessing strictly in the required order:
1. Removed duplicate rows (0 removed).
2. Removed rows with missing target values (0 removed).
3. Imputed missing numerical values with feature medians (0 imputed, since none were missing).
4. Dropped ID columns, one-hot encoded categorical inputs (`Geography`, `Gender`), and ensured integer target labels (`Exited`).
- Saved clean data into `milestone 1/Hloz_M1_Cleaned.csv` (10,000 rows × 14 columns).

### Step 6: Creation of Submission Deliverables
- Built and pre-executed `Hloz_M1.ipynb` with embedded plots and written student-level observations under every figure.
- Compiled `Hloz_M1_Report.pdf` strictly bounded to **6 pages maximum**.
- Initialized Git, committed all assets, and pushed to GitHub.

---

## 3. The 3 Official Submission Deliverables

These three files inside `milestone 1/` are the exact files required for submission:

| Deliverable File | Description | Status |
| :--- | :--- | :---: |
| **`Hloz_M1_Report.pdf`** | A 6-page academic summary report covering dataset description, descriptive statistics table, plots, observations, preprocessing summary, and conclusion. | ✅ Ready |
| **`Hloz_M1.ipynb`** | A complete, fully executed Jupyter Notebook with clean code, line-by-line comments, pre-baked cell outputs, and written observations under each plot group. | ✅ Ready |
| **`Hloz_M1_Cleaned.csv`** | The final preprocessed and encoded 10,000 × 14 dataset, ready for machine learning in Milestone 2. | ✅ Ready |

---

## 4. Dataset & Statistical Findings

### Key Distributions
- **`CreditScore`:** Normal/Gaussian bell curve centered at ~650, capped at 850.
- **`Age`:** Positively skewed with a peak between 30–40 years and outliers up to 92.
- **`Balance`:** Zero-inflated bimodal distribution; **36.17% of customers have exactly $0.00 balance**, while customers with money have a bell curve around $120,000.
- **`EstimatedSalary`:** Strictly uniform distribution across [0, 200,000].

### Key Business & Signal Insights
- **Age is the strongest churn predictor:** The median age of churned customers is **~45 years**, compared to **~36 years** for retained customers.
- **Credit Score does NOT separate churners:** The distribution of credit scores is almost identical between customers who stayed and left.
- **Low Multicollinearity:** All feature correlations are near 0, with the only notable correlation being `Balance` vs `NumOfProducts` ($r = -0.30$).

---

## 5. The Basic Preprocessing Pipeline

The original raw dataset (`data set/Churn_Modelling.csv`) was left untouched. Preprocessing was performed to produce `Hloz_M1_Cleaned.csv`:

```python
# 1. Drop duplicate rows
cleaned_df = df.drop_duplicates()

# 2. Drop rows where target is missing
cleaned_df = cleaned_df.dropna(subset=['Exited'])

# 3. Fill missing numerical values with feature median
for col in numerical_features:
    if cleaned_df[col].isnull().sum() > 0:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())

# Drop non-predictive ID columns
cleaned_df = cleaned_df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# 4. One-hot encode categorical features & integer encode target
cleaned_df['Exited'] = cleaned_df['Exited'].astype(int)
cleaned_df = pd.get_dummies(cleaned_df, columns=['Geography', 'Gender'], dtype=int)

# Export cleaned dataset
cleaned_df.to_csv("Hloz_M1_Cleaned.csv", index=False)
```

### Resulting Clean Matrix (10,000 rows × 14 columns):
- Numerical features: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.
- Target: `Exited` (`0` = Stayed, `1` = Left).
- One-hot binary features: `Geography_France`, `Geography_Germany`, `Geography_Spain`, `Gender_Female`, `Gender_Male`.

---

## 6. How Everything Works & How to Run it Manually

If you or your teammates want to run or re-generate any part of the project:

### Running the Notebook Interactively:
1. Open `milestone 1/Hloz_M1.ipynb` in Antigravity IDE, VS Code, or JupyterLab.
2. Select the Python 3.12 kernel.
3. Click **"Run All"** to execute every cell sequentially.

### Re-running Scripts from Terminal:
Open PowerShell in `e:\University\random signals\milestone 1` and run:

```powershell
# Re-generate all static plot images and Hloz_M1_Cleaned.csv
python generate_assets.py

# Re-build and pre-execute Hloz_M1.ipynb
python create_notebook.py

# Re-compile Hloz_M1_Report.pdf (strictly 6 pages)
python create_report.py
```

---

## 7. Repository File Map

```
e:\University\random signals\
│
├── Milestone 0.pdf                    <- Dataset selection specification
├── milestone1.pdf                     <- Milestone 1 EDA rubric & guidelines
├── README.md                          <- Main repository README with links & team info
├── EXPLANATION_AND_WALKTHROUGH.md     <- THIS FILE: Comprehensive project guide
├── .gitignore                         <- Git ignore rules (caches, temp files)
│
├── data set/
│   ├── Churn_Modelling.csv            <- Raw untouched dataset (10,000 rows, 14 cols)
│   └── link.txt                       <- Kaggle source link
│
└── milestone 1/
    ├── README.md                      <- Milestone 1 overview and guidelines
    ├── Hloz_M1.ipynb                  <- DELIVERABLE 1: Executed Jupyter Notebook
    ├── Hloz_M1_Cleaned.csv            <- DELIVERABLE 2: Cleaned & encoded dataset
    ├── Hloz_M1_Report.pdf             <- DELIVERABLE 3: 6-page summary PDF report
    ├── generate_assets.py             <- Script to generate plots and cleaned CSV
    ├── create_notebook.py             <- Script to build and run Hloz_M1.ipynb
    ├── create_report.py               <- Script to build Hloz_M1_Report.pdf
    └── figures/                       <- High-resolution plots used in report
        ├── target_distribution.png
        ├── missing_values.png
        ├── numerical_pdfs.png
        ├── numerical_cdfs.png
        ├── correlation_heatmap.png
        └── boxplots_key_features.png
```
