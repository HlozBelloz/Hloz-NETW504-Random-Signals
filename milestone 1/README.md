# NETW504 - Random Signals and Noise: Milestone 1 (EDA)
**Instructor:** Prof. Talal Elshabrawy  
**Team Name:** Hloz  
**Team Members:**  
- Mazen Mohamed Hamdy Altelbany (ID: 64-12371)  
- Malak Sherif Mohamed (ID: 64-10784)  
- Suhad Eyhab Rasheed (ID: 64-31506)  
- Sama Ismael Ahel (ID: 64-19880)  
**Workspace:** `E:\University\random signals\`

---

## 📌 Context & Project Overview

This directory contains the files, datasets, and deliverables for Milestone 1 of the **Random Signals and Noise (NETW504)** course.

### Primary Goals
1. Perform Exploratory Data Analysis (EDA) on an approved Kaggle classification dataset with $\ge 10,000$ samples and 2 to 4 target classes.
2. Produce all required statistical analyses, PDFs, CDFs, and correlation heatmaps.
3. Execute strict 4-step basic preprocessing.
4. Generate the final 3 deliverables:
   - `TeamName_M1.ipynb`: Executable Jupyter Notebook with all cells, outputs, and 1–3 sentence observations under every plot group.
   - `TeamName_M1_Cleaned.csv`: Final preprocessed and encoded CSV.
   - `TeamName_M1_Report.pdf`: Concise summary report (maximum 6 pages).

---

## 📊 Dataset Profile

- **Source:** [Kaggle - Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)
- **Local Path:** `E:\University\random signals\data set\Churn_Modelling.csv`
- **Total Records:** 10,000 rows
- **Total Columns:** 14 columns
- **Single Row Representation:** A single bank customer's demographic profile, account balance, credit metrics, activity status, and whether they retained or churned from the bank.

### Target Variable
- **Target Name:** `Exited`
- **Classes:** Binary classification (2 classes)
  - `0` (Customer stayed / Retained): 7,963 samples (~79.63%)
  - `1` (Customer left / Churned): 2,037 samples (~20.37%)
- **Class Balance:** Moderately imbalanced (~80% retained vs ~20% churned).

### Feature Breakdown
- **ID-like Columns (To Drop):**
  - `RowNumber` (Row index)
  - `CustomerId` (Unique customer account ID)
  - `Surname` (Customer surname / personal identifier)
- **Numerical Input Features (8):**
  - `CreditScore`: Customer's credit score (integer)
  - `Age`: Customer's age in years (integer)
  - `Tenure`: Number of years customer has stayed with bank (integer)
  - `Balance`: Account balance in bank currency (float)
  - `NumOfProducts`: Number of bank products used (integer: 1 to 4)
  - `HasCrCard`: Credit card ownership indicator (binary: 0 or 1)
  - `IsActiveMember`: Active bank member indicator (binary: 0 or 1)
  - `EstimatedSalary`: Estimated annual salary (float)
- **Categorical Input Features (2):**
  - `Geography`: Country of residence (`France`, `Spain`, `Germany`)
  - `Gender`: Biological sex (`Female`, `Male`)

---

## 🛠️ Required Tasks & Workflow Rules

### 1. Initial Inspection & Quality Checks
- Check shapes, types, and print first 5 rows.
- Count exact duplicate rows.
- Count and percentage of missing values per column.
- Numerical descriptive statistics: `count`, `mean`, `std`, `min`, `median`, `max`.
- Document observations on quality, balance, and outliers.

### 2. Mandatory Plots (Every plot group must have a 1–3 sentence observation)
- **Target Distribution:** Bar / count plot showing distribution of `Exited`.
- **Missing Values:** Bar plot showing missing value counts across all columns prior to cleaning.
- **Estimated PDF (Every Numerical Feature):** Normalized histogram (`density=True`) with KDE curve for each of the 8 numerical features.
- **Empirical CDF (Every Numerical Feature):** Empirical Cumulative Distribution Function plot for each of the 8 numerical features.
- **Correlation Heatmap:** Correlation matrix heatmap of numerical features after basic preprocessing.

### 3. Basic Preprocessing Pipeline (STRICT ORDER)
1. **Drop duplicate rows** (if any).
2. **Drop rows with missing target** values.
3. **Impute missing numerical values** using feature median.
4. **Encode features:**
   - One-hot encode categorical inputs (`Geography`, `Gender`).
   - Encode target classes as integer labels (`0` and `1`).
   - Keep and report the encoding dictionary / mapping.
*Note: Do not overwrite the original DataFrame early; keep both original (for EDA) and preprocessed (for ML ready export).*

---

## 💻 Environment & Tooling
- **IDE:** Antigravity IDE (VS Code based, runs interactive `.ipynb` natively).
- **Python Environment:** Python 3.12 (`C:\Users\Mazen\AppData\Local\Programs\Python\Python312\python.exe`).
- **Required Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `jupyter`.
- **Vault Sync:** Cross-referenced in Obsidian vault at `D:\Obsidian Volts\Hloz\Hloz\01 - Projects\Project - Random Signals Milestone 1.md`.
