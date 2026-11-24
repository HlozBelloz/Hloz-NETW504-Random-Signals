"""
build_explanation_pdfs.py
=========================
NETW504 Random Signals and Noise - Milestone 2
Team Name: Hloz
Author: Mazen Mohamed Hamdy Altelbany (ID: 64-12371) & Team Hloz

Generates 3 comprehensive, professional PDF explanation guides:
1. Hloz_M2_Code_Explanation.pdf             (Line-by-line notebook walkthrough)
2. Hloz_M2_Project_and_Report_Walkthrough.pdf (Complete project & report guide)
3. Hloz_M2_Evaluation_Questions_and_Answers.pdf (Professor oral defense preparation)
"""

import os
import json
import pymupdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
M2_SUBMIT_DIR = os.path.join(BASE_DIR, "..", "m2 submit")
JSON_PATH = os.path.join(BASE_DIR, "summary_data.json")

with open(JSON_PATH, "r") as f:
    summary_data = json.load(f)

# Reusable Numbered Canvas for professional running header and footer
class NumberedCanvasBuilder:
    def __init__(self, doc_title):
        self.doc_title = doc_title

    def __call__(self, *args, **kwargs):
        title = self.doc_title
        class DynamicNumberedCanvas(canvas.Canvas):
            def __init__(self, *c_args, **c_kwargs):
                super().__init__(*c_args, **c_kwargs)
                self._saved_page_states = []

            def showPage(self):
                self._saved_page_states.append(dict(self.__dict__))
                self._startPage()

            def save(self):
                num_pages = len(self._saved_page_states)
                for state in self._saved_page_states:
                    self.__dict__.update(state)
                    self.draw_page_decorations(num_pages)
                    super().showPage()
                super().save()

            def draw_page_decorations(self, page_count):
                self.saveState()
                self.setFont("Helvetica", 8)
                self.setFillColor(colors.HexColor("#555555"))
                
                # Header (pages 2+)
                if self._pageNumber > 1:
                    self.drawString(54, letter[1] - 36, f"NETW504 Random Signals & Noise — {title} | Team Hloz")
                    self.setStrokeColor(colors.HexColor("#dddddd"))
                    self.setLineWidth(0.5)
                    self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
                
                # Footer (all pages)
                page_text = f"Page {self._pageNumber} of {page_count}"
                self.drawRightString(letter[0] - 54, 30, page_text)
                self.drawString(54, 30, "Team Hloz — NETW504 Fall 2026 — Prof. Talal Elshabrawy")
                self.setStrokeColor(colors.HexColor("#dddddd"))
                self.setLineWidth(0.5)
                self.line(54, 42, letter[0] - 54, 42)
                
                self.restoreState()
        return DynamicNumberedCanvas(*args, **kwargs)

def get_custom_styles():
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2b6cb0"),
        spaceAfter=6
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#333333"),
        spaceAfter=8
    )
    
    h1_style = ParagraphStyle(
        'H1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1a365d"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#2b6cb0"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#222222"),
        spaceAfter=5
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a")
    )

    code_box = ParagraphStyle(
        'CodeBox',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1e293b"),
        backColor=colors.HexColor("#f8fafc"),
        borderColor=colors.HexColor("#cbd5e1"),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=6
    )

    callout_box = ParagraphStyle(
        'CalloutBox',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e3a8a"),
        backColor=colors.HexColor("#eff6ff"),
        borderColor=colors.HexColor("#93c5fd"),
        borderWidth=0.8,
        borderPadding=6,
        spaceAfter=6
    )

    q_style = ParagraphStyle(
        'Question_Style',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor("#1a365d"),
        spaceBefore=8,
        spaceAfter=2,
        keepWithNext=True
    )

    ans_style = ParagraphStyle(
        'Answer_Style',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#222222")
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    return {
        'title': title_style,
        'subtitle': subtitle_style,
        'meta': meta_style,
        'h1': h1_style,
        'h2': h2_style,
        'body': body_style,
        'code': code_style,
        'code_box': code_box,
        'callout': callout_box,
        'question': q_style,
        'answer': ans_style,
        'table_cell': table_cell,
        'table_header': table_header
    }

def make_header_banner(title_text, subtitle_text, styles):
    story = []
    story.append(Paragraph("NETW504: Random Signals and Noise — GUC Fall 2026", styles['subtitle']))
    story.append(Paragraph(title_text, styles['title']))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a365d"), spaceBefore=2, spaceAfter=6))
    
    meta_text = f"""
    <b>Instructor:</b> Prof. Talal Elshabrawy &nbsp;|&nbsp; <b>Team Name:</b> Hloz<br/>
    <b>Team Members:</b> Mazen Mohamed Hamdy Altelbany (64-12371), Malak Sherif Mohamed (64-10784), Suhad Eyhab Rasheed (64-31506), Sama Ismael Ahel (64-19880)<br/>
    <b>Topic:</b> {subtitle_text}
    """
    story.append(Paragraph(meta_text, styles['meta']))
    return story

# ==============================================================================
# PDF 1: NOTEBOOK CODE EXPLANATION GUIDE
# ==============================================================================
def generate_code_explanation_pdf(output_path):
    print(f"Generating Code Explanation PDF at: {output_path}")
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    styles = get_custom_styles()
    story = make_header_banner(
        "Milestone 2: Jupyter Notebook Code Walkthrough",
        "Cell-by-Cell Pedagogical Code Explanation & Mathematical Mapping (`Hloz_M2.ipynb`)",
        styles
    )

    story.append(Paragraph("1. Overview of the Notebook Architecture", styles['h1']))
    story.append(Paragraph(
        "The Jupyter Notebook <code>Hloz_M2.ipynb</code> is organized into 6 structured sections comprising 16 code cells. "
        "Every single cell was authored with strict adherence to data hygiene principles (zero data leakage) and includes rich, "
        "descriptive variable names and inline comments. Below is a comprehensive breakdown of every functional component.",
        styles['body']
    ))

    # Cell 1
    story.append(Paragraph("Cell 1: Core Library Imports & Dataset Loading", styles['h2']))
    story.append(Paragraph(
        "<b>Purpose:</b> Initializes the numerical computing, visualization, and machine learning environments, loads the raw bank churn CSV, "
        "and strips away non-predictive database keys.",
        styles['body']
    ))
    code_cell_1 = (
        "# Core data structures and mathematical routines<br/>"
        "import numpy as np &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Fast matrix operations and vectorized Gaussian densities<br/>"
        "import pandas as pd &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Tabular DataFrame manipulation and indexing<br/>"
        "# Plotting and visual inspection<br/>"
        "import matplotlib.pyplot as plt &nbsp;# Low-level figure canvas, subplots, and formatting<br/>"
        "import seaborn as sns &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Statistical theme styling and heatmaps<br/>"
        "# Scikit-learn pipelines & classifiers<br/>"
        "from sklearn.model_selection import train_test_split # Stratified partitioning<br/>"
        "from sklearn.impute import SimpleImputer &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Imputation of missing values<br/>"
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder # Scaling & dummy encoding<br/>"
        "from sklearn.compose import ColumnTransformer &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Feature-specific pipeline dispatch<br/>"
        "from sklearn.naive_bayes import GaussianNB &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Gaussian Naive Bayes benchmark<br/>"
        "from sklearn.tree import DecisionTreeClassifier &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Conventional non-linear tree<br/>"
        "from sklearn.linear_model import LogisticRegression &nbsp;&nbsp;# Conventional linear model<br/>"
        "# Load data & drop ID columns<br/>"
        "df_raw = pd.read_csv('../data set/Churn_Modelling.csv')<br/>"
        "df = df_raw.drop(columns=['RowNumber', 'CustomerId', 'Surname'])"
    )
    story.append(Paragraph(code_cell_1, styles['code_box']))
    story.append(Paragraph(
        "<b>Why drop ID columns?</b> <code>RowNumber</code>, <code>CustomerId</code>, and <code>Surname</code> are arbitrary primary keys. "
        "If left in the dataset, flexible classifiers could memorize customer IDs or surname strings, causing catastrophic overfitting and false signals.",
        styles['callout']
    ))

    # Cell 2
    story.append(Paragraph("Cell 2: Two-Stage Stratified Train / Validation / Test Splitting", styles['h2']))
    story.append(Paragraph(
        "<b>Purpose:</b> Divides the 10,000 samples into exactly 70% Training (7,000), 15% Validation (1,500), and 15% Test (1,500) sets "
        "while enforcing identical class ratios via stratification.",
        styles['body']
    ))
    code_cell_2 = (
        "# Stage 1: Hold out 30% for validation and test combined (70% train)<br/>"
        "X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;X, y, test_size=0.30, random_state=42, stratify=y<br/>"
        ")<br/>"
        "# Stage 2: Split the 30% holdout equally into 15% validation and 15% test<br/>"
        "X_val_raw, X_test_raw, y_val, y_test = train_test_split(<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;X_temp_raw, y_temp, test_size=0.50, random_state=42, stratify=y_temp<br/>"
        ")"
    )
    story.append(Paragraph(code_cell_2, styles['code_box']))
    story.append(Paragraph(
        "<b>Key Technical Details:</b><br/>"
        "• <code>stratify=y</code>: Guarantees that the ~79.6% stayed vs ~20.4% churned proportion is exactly maintained in all three splits.<br/>"
        "• <code>random_state=42</code>: Seeds the pseudo-random generator for 100% deterministic reproducibility across runs.<br/>"
        "• <code>test_size=0.50</code> in Stage 2: 50% of the 30% holdout yields exactly 15% of the total dataset.",
        styles['body']
    ))

    # Cell 3
    story.append(Paragraph("Cell 3: Building & Fitting Preprocessing Pipeline on Training ONLY", styles['h2']))
    story.append(Paragraph(
        "<b>Purpose:</b> Constructs a clean transformation pipeline and fits it <b>exclusively on training data</b>, preventing data leakage.",
        styles['body']
    ))
    code_cell_3 = (
        "num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])<br/>"
        "cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;('ohe', OneHotEncoder(drop='first', sparse_output=False))])<br/>"
        "preprocessor = ColumnTransformer([('num', num_pipe, numeric_cols), ('cat', cat_pipe, categorical_cols)])<br/>"
        "# STRICT DATA HYGIENE: FIT ON TRAIN ONLY!<br/>"
        "preprocessor.fit(X_train_raw)<br/>"
        "# Transform splits separately<br/>"
        "X_train = preprocessor.transform(X_train_raw)<br/>"
        "X_val &nbsp;&nbsp;= preprocessor.transform(X_val_raw)<br/>"
        "X_test &nbsp;= preprocessor.transform(X_test_raw)"
    )
    story.append(Paragraph(code_cell_3, styles['code_box']))
    story.append(Paragraph(
        "<b>The Data Leakage Rule:</b> Fitting <code>StandardScaler</code> on the entire dataset would expose the test set's mean (&mu;) and standard deviation (&sigma;) "
        "to the training process. By fitting on <code>X_train_raw</code> only, validation and test remain strictly out-of-sample unseen data.",
        styles['callout']
    ))

    story.append(PageBreak())

    # Cell 4 & 5
    story.append(Paragraph("Cell 4 & 5: Empirical Priors & Class Likelihood Parameters", styles['h1']))
    story.append(Paragraph(
        "<b>Purpose:</b> Computes empirical prior probabilities <i>P(Y=k)</i> and class-conditional Gaussian parameters (mean vector &mu;<sub>k</sub> "
        "and variance vector &sigma;<sub>k</sub><sup>2</sup>) directly from the training data.",
        styles['body']
    ))
    code_cell_4_5 = (
        "# Cell 4: Empirical Class Priors from Training Counts (N=7,000)<br/>"
        "prior_0 = (y_train == 0).sum() / len(y_train) # 5,574 / 7,000 = 0.7963 (79.63%)<br/>"
        "prior_1 = (y_train == 1).sum() / len(y_train) # 1,426 / 7,000 = 0.2037 (20.37%)<br/>"
        "# Cell 5: Gaussian Likelihood Parameters with Variance Smoothing<br/>"
        "var_smoothing = 1e-9 * np.max(np.var(X_train, axis=0)) # Numerical stabilizer<br/>"
        "for k in [0, 1]:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;X_k = X_train[y_train.values == k]<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;means[k] = np.mean(X_k, axis=0)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;variances[k] = np.var(X_k, axis=0) + var_smoothing"
    )
    story.append(Paragraph(code_cell_4_5, styles['code_box']))
    story.append(Paragraph(
        "<b>Variance Smoothing Explained:</b> If a feature has near-zero variance within a class, the Gaussian denominator "
        "&radic;(2&pi;&sigma;<sup>2</sup>) evaluates to 0, producing a fatal division-by-zero runtime error. Adding "
        "&epsilon; = 10<sup>-9</sup> &times; &sigma;<sub>max</sub><sup>2</sup> matches scikit-learn's exact numerical stability formula.",
        styles['body']
    ))

    # Cell 6 & 7
    story.append(Paragraph("Cell 6 & 7: Direct Bayesian Calculation & Posteriors Table Export", styles['h2']))
    story.append(Paragraph(
        "<b>Purpose:</b> Directly evaluates Bayes' Theorem from scratch using the log-sum-exp trick, verifies that row probabilities sum to 1.0, "
        "and exports the full 10,000-row posterior table.",
        styles['body']
    ))
    code_cell_6 = (
        "def compute_bayesian_posteriors(X_matrix):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;log_scores = np.zeros((len(X_matrix), 2))<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;for k in [0, 1]:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;log_prior = np.log(priors[k])<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;log_liks = -0.5*np.log(2*np.pi*variances[k]) - 0.5*((X_matrix - means[k])**2)/variances[k]<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;log_scores[:, k] = log_prior + np.sum(log_liks, axis=1)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;# Stable Softmax (Log-Sum-Exp trick to prevent underflow)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;max_log = np.max(log_scores, axis=1, keepdims=True)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;exp_scores = np.exp(log_scores - max_log)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;posteriors = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;predictions = np.argmax(posteriors, axis=1)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;return posteriors, predictions"
    )
    story.append(Paragraph(code_cell_6, styles['code_box']))
    story.append(Paragraph(
        "<b>The Log-Sum-Exp Technique:</b> When multiplying 11 Gaussian densities, floating-point numbers can underflow to 0.0. "
        "Computing in log space and subtracting <code>max_log</code> before exponentiating prevents floating-point underflow and overflow completely.",
        styles['callout']
    ))

    # Cell 8 & 9
    story.append(Paragraph("Cell 8 & 9: 20 Test Samples & Worked Bayes Calculation", styles['h2']))
    story.append(Paragraph(
        "<b>Purpose:</b> Selects 20 diverse test samples and provides a granular, feature-by-feature worked derivation for Customer 5750.",
        styles['body']
    ))
    code_cell_9 = (
        "# For each of the 11 features, compute likelihood under Y=0 and Y=1<br/>"
        "lik0 = (1.0 / np.sqrt(2 * np.pi * v0)) * np.exp(-0.5 * ((x_j - m0)**2) / v0)<br/>"
        "lik1 = (1.0 / np.sqrt(2 * np.pi * v1)) * np.exp(-0.5 * ((x_j - m1)**2) / v1)<br/>"
        "likelihood_ratio = lik1 / lik0 # Values > 1 indicate evidence pulling to Churn<br/>"
        "# Aggregate with prior odds to compute posterior probability P(Y=1 | X) = 83.61%"
    )
    story.append(Paragraph(code_cell_9, styles['code_box']))

    # Cell 10 & 11
    story.append(Paragraph("Cell 10 & 11: GaussianNB Benchmark & Equivalence Verification", styles['h2']))
    story.append(Paragraph(
        "<b>Purpose:</b> Trains scikit-learn's <code>GaussianNB</code> and proves machine-precision identity with scratch Bayes.",
        styles['body']
    ))
    code_cell_10 = (
        "gnb = GaussianNB()<br/>"
        "gnb.fit(X_train, y_train)<br/>"
        "gnb_probs = gnb.predict_proba(X_test)<br/>"
        "# Verification against manual implementation<br/>"
        "max_diff = np.max(np.abs(post_test - gnb_probs)) # Evaluates to 2.00e-15!"
    )
    story.append(Paragraph(code_cell_10, styles['code_box']))

    # Cell 12 & 13
    story.append(Paragraph("Cell 12 & 13: Conventional ML Validation Tuning (DT & LR)", styles['h2']))
    story.append(Paragraph(
        "<b>Purpose:</b> Uses the independent Validation Set to select optimal hyperparameters (tree depth and regularization strength).",
        styles['body']
    ))
    code_cell_12_13 = (
        "# Decision Tree Tuning: evaluate max_depth in [2, 15] on Validation Set<br/>"
        "best_tree = DecisionTreeClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=10)<br/>"
        "# Logistic Regression Tuning: evaluate C in [0.001, 100.0] on Validation Set<br/>"
        "best_lr = LogisticRegression(C=1.0, max_iter=1000)"
    )
    story.append(Paragraph(code_cell_12_13, styles['code_box']))

    # Cell 14, 15, 16
    story.append(Paragraph("Cell 14, 15 & 16: Test Set Evaluation & Comparison", styles['h2']))
    story.append(Paragraph(
        "<b>Purpose:</b> Evaluates all 4 models on the untouched 1,500 test samples, outputs the comparison table, labeled confusion matrices, "
        "and metric bar charts.",
        styles['body']
    ))

    doc.build(story, canvasmaker=NumberedCanvasBuilder("Notebook Code Explanation Guide"))
    print(f"Code Explanation PDF successfully built: {output_path}")

# ==============================================================================
# PDF 2: WHOLE PROJECT & REPORT WALKTHROUGH
# ==============================================================================
def generate_project_and_report_pdf(output_path):
    print(f"Generating Project and Report Walkthrough PDF at: {output_path}")
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    styles = get_custom_styles()
    story = make_header_banner(
        "Milestone 2: Project & Report Comprehensive Walkthrough",
        "Complete System Architecture, Statistical Derivations, and Page-by-Page Report Breakdown",
        styles
    )

    story.append(Paragraph("1. Executive Project Context & Random Signals Framing", styles['h1']))
    story.append(Paragraph(
        "In the <b>NETW504 Random Signals and Noise</b> curriculum, machine learning classification is studied through the rigorous lens of "
        "<b>Statistical Hypothesis Testing and Bayesian Decision Theory</b>. Predicting whether a customer will stay (<i>X</i>H_0<i>X</i>) or churn (<i>X</i>H_1<i>X</i>) "
        "mirrors signal detection in communication channels, where a noisy observation vector <i>X</i>X<i>X</i> must be mapped to a discrete transmitted state <i>X</i>Y<i>X</i>.",
        styles['body']
    ))
    story.append(Paragraph(
        "<b>Key Lessons Carried Over from Milestone 1 (EDA):</b><br/>"
        "• <i>Class Imbalance:</i> ~79.6% stayed vs ~20.4% churned. This 4-to-1 ratio imposes a severe prior bias favoring retention.<br/>"
        "• <i>Feature Distributions:</i> Credit score is Gaussian-like, age is right-skewed, salary is uniform, and balance is zero-inflated (36% zeros).<br/>"
        "• <i>Correlations:</i> Balance and NumOfProducts exhibit a negative correlation of -0.304, which directly violates the conditional independence assumption.",
        styles['body']
    ))

    story.append(Paragraph("2. Page-by-Page Walkthrough of `Hloz_M2_Report.pdf`", styles['h1']))

    # Page 1 breakdown
    story.append(Paragraph("Page 1: Title, Data Profile, Partitioning & Preprocessing", styles['h2']))
    story.append(Paragraph(
        "• <b>Header & Metadata:</b> Formally documents course details, Team Hloz member names and student IDs, and project targets.<br/>"
        "• <b>Dataset Preparation:</b> Details the dropping of ID keys (`RowNumber`, `CustomerId`, `Surname`) and defines target mapping (0: Stayed, 1: Churned).<br/>"
        "• <b>Table 1 (Dataset Partitioning):</b> Shows exact counts for Train (7,000), Val (1,500), and Test (1,500) sets, demonstrating that stratification maintained the exact 20.37% churn rate across all splits.<br/>"
        "• <b>Figure 1 (Partitioning & Priors):</b> High-resolution chart visually depicting split sizes and the empirical training prior probabilities.",
        styles['body']
    ))

    # Page 2 breakdown
    story.append(Paragraph("Page 2: Bayesian Estimation from Scratch & 20-Sample Study", styles['h2']))
    story.append(Paragraph(
        "• <b>Empirical Priors:</b> Establishes <i>X</i>P(Y=0) = 0.7963<i>X</i> and <i>X</i>P(Y=1) = 0.2037<i>X</i>.<br/>"
        "• <b>Rebalancing Discussion:</b> Explains why artificial rebalancing (e.g. SMOTE to 50/50) distorts production priors, requiring Bayesian odds correction.<br/>"
        "• <b>Mathematical Derivation:</b> Formulates Bayes' Theorem in log-space with univariate Gaussian likelihoods.<br/>"
        "• <b>Dr. Note Analysis:</b> Directly addresses Dr. Elshabrawy's prompt note on conditional independence violations and evidence double-counting.<br/>"
        "• <b>Table 2 (20 Test Samples):</b> Presents 10 actual stayed and 10 actual churned test samples with raw features, priors, posteriors, and predictions.",
        styles['body']
    ))

    story.append(PageBreak())

    # Page 3 breakdown
    story.append(Paragraph("Page 3: Step-by-Step Worked Bayes Example (Sample 5750)", styles['h1']))
    story.append(Paragraph(
        "• <b>Customer Profile:</b> Customer 5750 (Woronoff) has 3 products, resides in Germany, is inactive, and churned (<i>Y</i>=1).<br/>"
        "• <b>Likelihood Ratio Analysis:</b> Explains how individual feature likelihood ratios <i>LR</i><sub>j</sub> = <i>p</i>(<i>x</i><sub>j</sub> | <i>Y</i>=1) / <i>p</i>(<i>x</i><sub>j</sub> | <i>Y</i>=0) act as multiplicative weights updating prior odds.<br/>"
        "• <b>Figure 2 (Log-Likelihood Contributions):</b> Horizontal bar chart illustrating the positive and negative evidence forces.<br/>"
        "• <b>Final Posterior Update:</b> Proves how strong feature signals (<i>LR</i><sub>total</sub> &approx; 16.1) overturned the prior to produce <i>P</i>(<i>Y</i>=1 | <i>X</i>) = 83.61%.",
        styles['body']
    ))

    # Page 4 breakdown
    story.append(Paragraph("Page 4: Bayesian Machine Learning (GaussianNB)", styles['h2']))
    story.append(Paragraph(
        "• <b>Model Justification:</b> Explains why GaussianNB is suited for standardized continuous features over BernoulliNB or MultinomialNB.<br/>"
        "• <b>Numerical Identity Verification:</b> Confirms that manual scratch Bayes matches `GaussianNB.predict_proba()` within 2.00 &times; 10<sup>-15</sup>.<br/>"
        "• <b>Figure 3 (20-Sample Probability Distribution):</b> Visualizes model-estimated probabilities with the 0.5 decision boundary line.",
        styles['body']
    ))

    # Page 5 breakdown
    story.append(Paragraph("Page 5: Conventional Machine Learning (Decision Tree & Logistic Regression)", styles['h2']))
    story.append(Paragraph(
        "• <b>Decision Tree Mechanics:</b> Recursive partitioning minimizing Gini Impurity: Gini = 1 - &sum; <i>p</i><sub>k</sub><sup>2</sup>.<br/>"
        "• <b>Logistic Regression Mechanics:</b> Linear log-odds passing through the sigmoid activation &sigma;(<i>z</i>) = 1 / (1 + <i>e</i><sup>-<i>z</i></sup>).<br/>"
        "• <b>Figure 4 (Validation Tuning Curves):</b> Shows how `max_depth = 8` and <i>C</i> = 1.0 were selected strictly using validation data.",
        styles['body']
    ))

    # Page 6 breakdown
    story.append(Paragraph("Page 6: Untouched Test Set Evaluation, Confusion Matrices & Conclusions", styles['h2']))
    story.append(Paragraph(
        "• <b>Table 3 (Performance Comparison):</b> Compares Accuracy, Macro Precision, Macro Recall, Macro F1, and Class F1 across all models.<br/>"
        "• <b>Figure 5 (Confusion Matrices):</b> 2x2 grid displaying counts and recall percentages for all 4 classifiers.<br/>"
        "• <b>Technical Insights:</b> Explains why Decision Tree achieved superior performance (Macro F1 = 0.7696) due to non-linear interaction modeling, "
        "and highlights why Accuracy (81.5% - 86.7%) is deceptive under class imbalance.",
        styles['body']
    ))

    story.append(Spacer(1, 4))
    story.append(Paragraph("3. Summary of Untouched Test Set Benchmark", styles['h1']))
    eval_table_data = [
        [Paragraph("Model Name", styles['table_header']), Paragraph("Accuracy", styles['table_header']), Paragraph("Macro Prec", styles['table_header']), Paragraph("Macro Rec", styles['table_header']), Paragraph("Macro F1", styles['table_header']), Paragraph("Class 0 F1", styles['table_header']), Paragraph("Class 1 F1", styles['table_header'])]
    ]
    for r in summary_data['eval_table']:
        eval_table_data.append([
            Paragraph(r['Model'], styles['table_cell']),
            Paragraph(f"{r['Accuracy']:.4f}", styles['table_cell']),
            Paragraph(f"{r['Macro Precision']:.4f}", styles['table_cell']),
            Paragraph(f"{r['Macro Recall']:.4f}", styles['table_cell']),
            Paragraph(f"<b>{r['Macro F1']:.4f}</b>", styles['table_cell']),
            Paragraph(f"{r['Class 0 F1 (Stayed)']:.4f}", styles['table_cell']),
            Paragraph(f"<b>{r['Class 1 F1 (Churned)']:.4f}</b>", styles['table_cell'])
        ])
    t_ev = Table(eval_table_data, colWidths=[1.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.9*inch, 0.9*inch])
    t_ev.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1a365d")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8f9fa")]),
    ]))
    story.append(t_ev)

    doc.build(story, canvasmaker=NumberedCanvasBuilder("Project & Report Walkthrough"))
    print(f"Project and Report Walkthrough PDF successfully built: {output_path}")

# ==============================================================================
# PDF 3: EVALUATION QUESTIONS AND ANSWERS (ORAL DEFENSE PREP)
# ==============================================================================
def generate_evaluation_qa_pdf(output_path):
    print(f"Generating Evaluation Q&A PDF at: {output_path}")
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    styles = get_custom_styles()
    story = make_header_banner(
        "Milestone 2: Oral Defense & Evaluation Preparation Guide",
        "Anticipated Professor Examination Questions, Technical Explanations, and Defense Answers",
        styles
    )

    story.append(Paragraph("How to Approach the Oral Evaluation", styles['h1']))
    story.append(Paragraph(
        "In NETW504, evaluations are designed to test your <b>conceptual depth</b>, not rote memorization. Prof. Talal Elshabrawy and the TAs "
        "will test whether you understand the physical meaning of the equations, the reason behind every line of code, and how to detect traps. "
        "Use the answers below to speak with authority, technical rigor, and clarity.",
        styles['body']
    ))

    # Category 1
    story.append(Paragraph("Category 1: Data Preparation, Splitting & Data Leakage", styles['h1']))

    story.append(Paragraph("Q1: Why did you split into Train, Validation, and Test BEFORE fitting the scalers and encoders?", styles['question']))
    story.append(Paragraph(
        "<b>Answer:</b> Fitting any transformer (like <code>StandardScaler</code> or <code>SimpleImputer</code>) on the full dataset before splitting "
        "is a critical statistical mistake called <b>data leakage</b>. Standard scaling computes the mean &mu; and standard deviation &sigma; of each feature. "
        "If computed across all 10,000 rows, the test set's distribution parameters leak into the training feature representation, giving artificially optimistic results. "
        "By strictly calling <code>fit(X_train)</code> on training data only and applying <code>transform()</code> to validation and test, our validation and test sets remain "
        "completely unseen out-of-sample data.",
        styles['answer']
    ))

    story.append(Paragraph("Q2: Why was stratification essential in your train/val/test split?", styles['question']))
    story.append(Paragraph(
        "<b>Answer:</b> Our dataset has a moderate class imbalance: ~79.6% stayed vs ~20.4% churned. In random sampling without stratification, small subsets "
        "(like our 15% validation or test sets) could randomly receive too few churners (e.g. 15% or 10%), artificially distorting class priors and performance metrics. "
        "Stratification guarantees that every split maintains the exact 79.6% / 20.4% ratio, ensuring statistical consistency across training, tuning, and evaluation.",
        styles['answer']
    ))

    story.append(Paragraph("Q3: Why did you use `drop='first'` in your OneHotEncoder?", styles['question']))
    story.append(Paragraph(
        "<b>Answer:</b> If a categorical feature has <i>K</i> categories and we create <i>K</i> binary dummy variables, their sum always equals 1.0. "
        "This creates perfect multicollinearity, known as the <b>dummy variable trap</b>, which leads to a singular covariance matrix that cannot be inverted in linear models. "
        "Dropping the first category (e.g. France for Geography, Female for Gender) leaves <i>K-1</i> linearly independent columns, where the omitted category serves as the reference baseline.",
        styles['answer']
    ))

    story.append(PageBreak())

    # Category 2
    story.append(Paragraph("Category 2: Bayesian Estimation & Random Signals", styles['h1']))

    story.append(Paragraph("Q4: Explain how you computed posterior probabilities directly from Bayes' Theorem.", styles['question']))
    story.append(Paragraph(
        "<b>Answer:</b> We formulated Bayes' Theorem directly:<br/>"
        "<font face='Helvetica-Bold' color='#1a365d'>score<sub>k</sub>(x) = P(Y=k) &times; &prod; p(x<sub>j</sub> | Y=k) &nbsp;;&nbsp; P(Y=k | x) = score<sub>k</sub>(x) / &sum; score<sub>c</sub>(x)</font><br/>"
        "We estimated the empirical priors <i>P(Y=k)</i> from training class counts (79.63% vs 20.37%). Under the Gaussian Naive Bayes assumption, each feature likelihood "
        "<i>p(x<sub>j</sub> | Y=k)</i> was modeled as a normal density <i>N(&mu;<sub>kj</sub>, &sigma;<sub>kj</sub><sup>2</sup>)</i> estimated from training data. "
        "To avoid floating-point underflow from multiplying 11 densities, we computed scores in log-space and applied the stable softmax (log-sum-exp trick) to normalize them.",
        styles['answer']
    ))

    story.append(Paragraph("Q5: What happens if you balance the training data using SMOTE or undersampling?", styles['question']))
    story.append(Paragraph(
        "<b>Answer:</b> Artificially rebalancing the training data forces the training prior to become <i>P(Y=0) = P(Y=1) = 0.50</i>. "
        "In Bayes' theorem, removing the prior penalty against the minority class shifts the decision boundary toward class 0, boosting recall for churners. "
        "However, in the real world, the true base rate is still 20%. An unadjusted 50/50 model deployed in production will output far too many False Positives. "
        "To deploy properly, model probabilities must be re-calibrated using Bayes' rule by multiplying by the true prior odds ratio: "
        "<i>P<sub>true</sub>(Y=k) / P<sub>train</sub>(Y=k)</i>.",
        styles['answer']
    ))

    story.append(Paragraph("Q6: Dr. Elshabrawy noted: 'This assumes conditional independence; highly correlated features may violate it.' What did you discover?", styles['question']))
    story.append(Paragraph(
        "<b>Answer:</b> The conditional independence assumption states that <i>p(x<sub>1</sub>, x<sub>2</sub> | Y) = p(x<sub>1</sub> | Y) p(x<sub>2</sub> | Y)</i>. "
        "In our EDA (Milestone 1), we observed a correlation of -0.304 between <code>Balance</code> and <code>NumOfProducts</code>. When two correlated features are present, "
        "Naive Bayes treats them as two completely separate independent pieces of evidence, effectively <b>double-counting the evidence</b>. This pushes posterior probabilities "
        "to extreme overconfidence (closer to 0.0 or 1.0). However, the MAP classification decision depends solely on the argmax ranking, which is why classification accuracy remains robust.",
        styles['answer']
    ))

    story.append(Paragraph("Q7: Walk through Customer 5750 in the worked Bayes example. Why did the posterior flip from 20% to 84%?", styles['question']))
    story.append(Paragraph(
        "<b>Answer:</b> Customer 5750 started with a prior odds ratio strongly favoring staying: <i>P(Y=1)/P(Y=0) = 0.2037 / 0.7963 &approx; 0.256</i>. "
        "However, their feature likelihood ratios heavily favored churn:<br/>"
        "• <code>NumOfProducts = 3</code> had a Likelihood Ratio of <b>6.498</b> (historical churn rate for 3 products is over 80%).<br/>"
        "• <code>Geography = Germany</code> had an LR of <b>2.561</b> (German customers churn at double the rate of French customers).<br/>"
        "• <code>IsActiveMember = 0</code> had an LR of <b>1.434</b>.<br/>"
        "When multiplied together, the cumulative Likelihood Ratio was &approx; 16.1. Multiplying prior odds (0.256) by evidence (16.1) gives posterior odds of &approx; 4.1, "
        "which normalizes to <b>P(Y=1 | X) = 83.61%</b>, correctly flipping the classification to Churned.",
        styles['answer']
    ))

    story.append(PageBreak())

    # Category 3 & 4
    story.append(Paragraph("Category 3: Model Mechanics & Evaluation Traps", styles['h1']))

    story.append(Paragraph("Q8: Why is Accuracy a misleading metric in this project?", styles['question']))
    story.append(Paragraph(
        "<b>Answer:</b> Because 79.67% of the test set customers stayed, a naive 'dummy' classifier that simply predicts 0 for every customer achieves an "
        "<b>Accuracy of 79.67%</b> despite having <b>0.0% Recall</b> on churners! For a bank, missing a churner means losing a client and their deposits. "
        "Therefore, we must use <b>Macro F1</b> and <b>Minority Recall</b>, which give equal weight to both classes regardless of class frequency.",
        styles['answer']
    ))

    story.append(Paragraph("Q9: Why did Decision Tree outperform Logistic Regression and Naive Bayes on this dataset?", styles['question']))
    story.append(Paragraph(
        "<b>Answer:</b> Bank customer churn is heavily driven by <b>non-linear feature interactions</b>. For example, older age alone is not fatal, but older age combined with "
        "having 3 products and living in Germany creates an extreme probability of churn. A Decision Tree creates orthogonal hyperplanes that easily isolate these complex multi-feature regions. "
        "In contrast, Logistic Regression fits a single linear hyperplane, which fails to separate non-linear combinations without manual polynomial interaction terms.",
        styles['answer']
    ))

    story.append(Paragraph("Q10: Why did you tune Decision Tree max_depth on the validation set rather than letting it grow unconstrained?", styles['question']))
    story.append(Paragraph(
        "<b>Answer:</b> An unconstrained decision tree splits until every leaf is pure, memorizing training noise and overfitting. On our training set, depth 15 achieved an F1 of 0.95, "
        "but validation F1 degraded. By evaluating depths on the independent validation set, we found that <code>max_depth = 8</code> provided the optimal peak in generalization (Validation Macro F1 = 0.7417).",
        styles['answer']
    ))

    story.append(Paragraph("Q11: How did you prove that your scratch Bayesian implementation was mathematically exact?", styles['question']))
    story.append(Paragraph(
        "<b>Answer:</b> We evaluated all 1,500 test samples using both our scratch Bayesian algorithm and scikit-learn's <code>GaussianNB.predict_proba()</code>. "
        "The maximum absolute difference across all samples and classes was <b>2.00 &times; 10<sup>-15</sup></b>, which is within IEEE 754 double-precision floating-point roundoff. "
        "This mathematically proves that our manual implementation of Bayes' Theorem, Gaussian likelihoods, and variance smoothing is identical to scikit-learn's underlying algorithm.",
        styles['answer']
    ))

    doc.build(story, canvasmaker=NumberedCanvasBuilder("Evaluation Defense Guide"))
    print(f"Evaluation Q&A PDF successfully built: {output_path}")

if __name__ == "__main__":
    pdf1_path = os.path.join(BASE_DIR, "Hloz_M2_Code_Explanation.pdf")
    pdf2_path = os.path.join(BASE_DIR, "Hloz_M2_Project_and_Report_Walkthrough.pdf")
    pdf3_path = os.path.join(BASE_DIR, "Hloz_M2_Evaluation_Questions_and_Answers.pdf")
    
    generate_code_explanation_pdf(pdf1_path)
    generate_project_and_report_pdf(pdf2_path)
    generate_evaluation_qa_pdf(pdf3_path)
    
    # Mirror into m2 submit
    os.makedirs(M2_SUBMIT_DIR, exist_ok=True)
    import shutil
    shutil.copy(pdf1_path, os.path.join(M2_SUBMIT_DIR, "Hloz_M2_Code_Explanation.pdf"))
    shutil.copy(pdf2_path, os.path.join(M2_SUBMIT_DIR, "Hloz_M2_Project_and_Report_Walkthrough.pdf"))
    shutil.copy(pdf3_path, os.path.join(M2_SUBMIT_DIR, "Hloz_M2_Evaluation_Questions_and_Answers.pdf"))
    print("All 3 explanation PDFs generated and copied to m2 submit successfully!")
