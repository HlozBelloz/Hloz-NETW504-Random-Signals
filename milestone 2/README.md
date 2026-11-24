# NETW504 - Random Signals and Noise: Milestone 2
**Bayesian Estimation and Machine Learning**  
**Instructor:** Prof. Talal Elshabrawy  
**Team Name:** Hloz  
**Team Members:**  
- Mazen Mohamed Hamdy Altelbany (ID: 64-12371)  
- Malak Sherif Mohamed (ID: 64-10784)  
- Suhad Eyhab Rasheed (ID: 64-31506)  
- Sama Ismael Ahel (ID: 64-19880)  
**Workspace:** `E:\University\random signals\`

---

## 📌 Milestone 2 Overview & Objectives

Milestone 2 explores the application of **Bayesian Decision Theory**, statistical estimation, and machine learning classifiers on the approved Milestone 1 dataset (**Bank Customer Churn Prediction**).

### Primary Objectives:
1. **Dataset Preparation:** Partition the dataset into stratified 70% Training, 15% Validation, and 15% Test splits. Fit all missing-value imputation, categorical encoding, and numerical scaling strictly on the training set to prevent data leakage.
2. **Bayesian Estimation from Scratch:** Compute empirical training class priors ($P(Y=0) \approx 79.6\%, P(Y=1) \approx 20.4\%$). Formulate Gaussian feature likelihoods under the conditional independence assumption. Directly calculate posterior probabilities $P(Y=k \mid X=x_i)$ for all 10,000 samples via Bayes' Theorem using stable log-sum-exp score normalization.
3. **Bayesian Machine Learning (`GaussianNB`):** Train scikit-learn's Gaussian Naive Bayes and verify that its probability outputs are mathematically identical to our scratch Bayesian estimator (maximum difference $\le 2 \times 10^{-15}$).
4. **Conventional Machine Learning:** Train and tune a **Decision Tree** (`max_depth = 8`) and **Logistic Regression** ($C = 1.0$) using the independent Validation Set without touching the test set.
5. **Untouched Test Set Evaluation:** Benchmark all models across Accuracy, Macro Precision, Macro Recall, Macro F1, and labeled Confusion Matrices. Conduct an in-depth analysis on class imbalance and the impact of conditional independence violations.
6. **Deliverables:**
   - `Hloz_M2.ipynb`: Fully executed Jupyter Notebook with saved outputs, plots, and thorough markdown and inline code explanations.
   - `Hloz_M2_Report.pdf`: Concise, professional 6-page academic report strictly within the 6-page limit.
   - `Hloz_M2_Bayesian_Posteriors.csv`: Full table of 10,000 samples with sample IDs, splits, actual classes, posteriors, and predicted classes.

---

## 📂 Deliverables Directory Map

```
E:\University\random signals\
├── milestone 2\
│   ├── figures\                          # High-resolution generated plots and page inspection PNGs
│   │   ├── fig1_splits_and_priors.png
│   │   ├── fig2_20_samples_probabilities.png
│   │   ├── fig3_worked_example_likelihoods.png
│   │   ├── fig4_validation_tuning.png
│   │   ├── fig5_confusion_matrices.png
│   │   └── fig6_model_comparison.png
│   ├── Hloz_M2.ipynb                     # Fully executed notebook with all outputs & code explanations
│   ├── Hloz_M2_Report.pdf                # Exactly 6-page professional PDF report
│   ├── Hloz_M2_Bayesian_Posteriors.csv   # Complete Bayesian posterior probabilities table (10,000 rows)
│   ├── Hloz_M2_Code_Explanation.pdf      # Dedicated PDF: Line-by-line explanation of every notebook code cell
│   ├── Hloz_M2_Project_and_Report_Walkthrough.pdf # Dedicated PDF: Complete project & report guide
│   ├── Hloz_M2_Evaluation_Questions_and_Answers.pdf # Dedicated PDF: Oral defense preparation Q&A guide
│   ├── build_explanation_pdfs.py         # Automated compiler for the 3 explanation PDFs
│   ├── generate_assets.py                # Asset calculation and figure generation script
│   ├── create_notebook.py                # Automated notebook builder & executor (via nbclient)
│   ├── create_report.py                  # Automated 6-page PDF compiler (via ReportLab)
│   ├── summary_data.json                 # Exported statistical metrics and evaluation tables
│   ├── EXPLANATION_AND_WALKTHROUGH_M2.md # Line-by-line code explanation & oral defense guide
│   └── README.md                         # This file
│
└── m2 submit\                            # Official submission folder
    ├── Hloz_M2.ipynb
    ├── Hloz_M2_Report.pdf
    ├── Hloz_M2_Bayesian_Posteriors.csv
    ├── Hloz_M2_Code_Explanation.pdf
    ├── Hloz_M2_Project_and_Report_Walkthrough.pdf
    └── Hloz_M2_Evaluation_Questions_and_Answers.pdf
```

---

## 📊 Summary of Test Set Performance (Untouched 1,500 Samples)

| Model Name | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Class 0 F1 (Stayed) | Class 1 F1 (Churned) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct Bayes (Scratch)** | 82.47% | 0.7365 | 0.6482 | 0.6722 | 0.8958 | 0.4486 |
| **Gaussian Naive Bayes** | 82.47% | 0.7365 | 0.6482 | 0.6722 | 0.8958 | 0.4486 |
| **Decision Tree (depth=8)** | **86.73%** | **0.8158** | **0.7421** | **0.7696** | **0.9197** | **0.6195** |
| **Logistic Regression (C=1.0)** | 81.53% | 0.7364 | 0.5874 | 0.6013 | 0.8934 | 0.3092 |

### Key Findings:
- **Decision Tree achieved the highest performance** (Macro F1 = 0.7696, Churn Recall = 58.0%) because it captures non-linear feature interactions (such as older customers in Germany with multiple products) that linear models cannot separate.
- **Accuracy is deceptive:** Trivial guessing of class 0 yields ~79.6% accuracy. Macro F1 and minority recall are the true indicators of utility in banking retention.
- **Mathematical Identity:** Direct Bayes from scratch matches `GaussianNB` within $2 \times 10^{-15}$.

