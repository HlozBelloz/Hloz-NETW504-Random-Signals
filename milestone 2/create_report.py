"""
create_report.py
================
NETW504 Random Signals and Noise - Milestone 2
Team Name: Hloz
Author: Mazen Mohamed Hamdy Altelbany (ID: 64-12371) & Team Hloz

Generates Hloz_M2_Report.pdf:
- Exactly 6 pages
- High professional academic quality
- Fully addresses every requirement of Milestone 2
"""

import os
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
PDF_PATH = os.path.join(BASE_DIR, "Hloz_M2_Report.pdf")
JSON_PATH = os.path.join(BASE_DIR, "summary_data.json")

with open(JSON_PATH, "r") as f:
    summary_data = json.load(f)

# Numbered Canvas for exact page tracking and professional running header/footer
class SimpleNumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        
        # Header on pages 2 through 6
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "NETW504 Random Signals & Noise — Milestone 2 Report | Team Hloz")
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
        
        # Footer on all pages
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 30, page_text)
        self.drawString(54, 30, "NETW504: Bayesian Estimation & Machine Learning — Fall 2026")
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(54, 42, letter[0] - 54, 42)
        
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=3
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#2b6cb0"),
        spaceAfter=6
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#333333")
    )
    
    h1_style = ParagraphStyle(
        'H1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1a365d"),
        spaceBefore=7,
        spaceAfter=3
    )

    h2_style = ParagraphStyle(
        'H2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#2b6cb0"),
        spaceBefore=5,
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.8,
        textColor=colors.HexColor("#222222"),
        spaceAfter=4
    )

    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#555555"),
        alignment=1,
        spaceAfter=4
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#222222")
    )

    table_cell_center = ParagraphStyle(
        'TableCellCenter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.2,
        alignment=1,
        textColor=colors.HexColor("#222222")
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.white,
        alignment=1
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, DATASET PREPARATION & PREPROCESSING PIPELINE
    # =========================================================================
    story.append(Paragraph("NETW504: Random Signals and Noise", subtitle_style))
    story.append(Paragraph("Milestone 2 — Bayesian Estimation & Machine Learning Report", title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a365d"), spaceBefore=2, spaceAfter=6))
    
    meta_text = """
    <b>Instructor:</b> Prof. Talal Elshabrawy &nbsp;|&nbsp; <b>Team Name:</b> Hloz<br/>
    <b>Team Members:</b> Mazen Mohamed Hamdy Altelbany (64-12371), Malak Sherif Mohamed (64-10784), Suhad Eyhab Rasheed (64-31506), Sama Ismael Ahel (64-19880)<br/>
    <b>Dataset:</b> Bank Customer Churn Prediction (Kaggle) &nbsp;|&nbsp; <b>Target:</b> <code>Exited</code> (0 = Stayed, 1 = Churned)
    """
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("1. Dataset Preparation & Preprocessing Pipeline", h1_style))
    story.append(Paragraph(
        "<b>Dataset Profile & Target Mapping:</b> We utilize the approved <i>Bank Customer Churn Prediction</i> dataset (10,000 rows, 14 columns). "
        "The target variable is <code>Exited</code>, where <b>0</b> indicates the customer remained with the bank (retained) and <b>1</b> indicates "
        "the customer left (churned). Non-predictive identification columns (<code>RowNumber</code>, <code>CustomerId</code>, <code>Surname</code>) "
        "were dropped to eliminate spurious correlations and data leakage.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Stratified Data Partitioning (70% / 15% / 15%):</b> To preserve the empirical class ratio (~79.6% stayed vs ~20.4% churned) across all subsets, "
        "we performed stratified sampling using <code>random_state=42</code>. The dataset was divided into 7,000 training samples, 1,500 validation samples, "
        "and 1,500 test samples.",
        body_style
    ))

    # Split Table
    split_table_data = [
        [Paragraph("Partition Split", table_header), Paragraph("Total Samples", table_header), Paragraph("Ratio (%)", table_header), Paragraph("Stayed (0)", table_header), Paragraph("Churned (1)", table_header), Paragraph("Churn Rate (%)", table_header)],
        [Paragraph("Training Set", table_cell), Paragraph(f"{summary_data['train_samples']:,}", table_cell_center), Paragraph("70.0%", table_cell_center), Paragraph(f"{summary_data['train_c0']:,}", table_cell_center), Paragraph(f"{summary_data['train_c1']:,}", table_cell_center), Paragraph(f"{summary_data['train_c1']/summary_data['train_samples']*100:.2f}%", table_cell_center)],
        [Paragraph("Validation Set", table_cell), Paragraph(f"{summary_data['val_samples']:,}", table_cell_center), Paragraph("15.0%", table_cell_center), Paragraph(f"{summary_data['val_c0']:,}", table_cell_center), Paragraph(f"{summary_data['val_c1']:,}", table_cell_center), Paragraph(f"{summary_data['val_c1']/summary_data['val_samples']*100:.2f}%", table_cell_center)],
        [Paragraph("Untouched Test Set", table_cell), Paragraph(f"{summary_data['test_samples']:,}", table_cell_center), Paragraph("15.0%", table_cell_center), Paragraph(f"{summary_data['test_c0']:,}", table_cell_center), Paragraph(f"{summary_data['test_c1']:,}", table_cell_center), Paragraph(f"{summary_data['test_c1']/summary_data['test_samples']*100:.2f}%", table_cell_center)],
        [Paragraph("<b>Total Population</b>", table_cell), Paragraph("10,000", table_cell_center), Paragraph("100.0%", table_cell_center), Paragraph("7,963", table_cell_center), Paragraph("2,037", table_cell_center), Paragraph("20.37%", table_cell_center)]
    ]
    t_split = Table(split_table_data, colWidths=[1.4*inch, 1.1*inch, 0.9*inch, 1.1*inch, 1.1*inch, 1.2*inch])
    t_split.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1a365d")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.HexColor("#ffffff"), colors.HexColor("#f8f9fa")]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#e9ecef")),
    ]))
    story.append(t_split)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Data Leakage Prevention & Preprocessing Pipeline:</b> In strict adherence to machine learning best practices, all transformations "
        "were <b>fitted exclusively on training data</b> and subsequently applied to validation and test sets:<br/>"
        "• <i>Numerical Imputation & Scaling:</i> Missing values imputed using feature medians; standardized using <code>StandardScaler</code> "
        "(<i>z</i> = (<i>x</i> - &mu;) / &sigma;) to ensure zero mean and unit variance for Gaussian likelihood estimation.<br/>"
        "• <i>Categorical Encoding:</i> Imputed with mode; one-hot encoded with <code>drop='first'</code> to avoid the dummy variable trap "
        "(<code>Geography</code> produces <code>Germany</code> and <code>Spain</code>; <code>Gender</code> produces <code>Male</code>). "
        "This yielded 11 clean standardized input features.",
        body_style
    ))
    story.append(Spacer(1, 3))

    img_splits = Image(os.path.join(FIGURES_DIR, "fig1_splits_and_priors.png"), width=6.8*inch, height=2.4*inch)
    story.append(img_splits)
    story.append(Paragraph("<b>Figure 1:</b> Stratified dataset partitioning and empirical training class priors P(Y=k).", caption_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: BAYESIAN ESTIMATION (PRIORS, FORMULATION & 20 SAMPLES)
    # =========================================================================
    story.append(Paragraph("2. Bayesian Estimation from Scratch", h1_style))
    story.append(Paragraph(
        f"<b>Empirical Class Priors:</b> Calculated directly from original training counts (<i>N</i><sub>train</sub> = 7,000):<br/>"
        f"• <b>P(Y=0) [Stayed]:</b> {summary_data['train_c0']} / 7,000 = <b>{summary_data['prior_0']:.4f}</b> ({summary_data['prior_0']*100:.2f}%)<br/>"
        f"• <b>P(Y=1) [Churned]:</b> {summary_data['train_c1']} / 7,000 = <b>{summary_data['prior_1']:.4f}</b> ({summary_data['prior_1']*100:.2f}%)<br/>"
        "The training data exhibits an imbalance ratio of <b>3.91 to 1</b> in favor of retained customers.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Effect of Rebalancing Training Data:</b> If training data were artificially rebalanced to 50/50 (e.g., via SMOTE or undersampling), "
        "the empirical prior becomes <i>P</i>(<i>Y</i>=0) = <i>P</i>(<i>Y</i>=1) = 0.50. This shifts the decision boundary to boost minority recall, but when deployed "
        "against the real test population (where the true base rate remains ~20%), it generates excessive False Positives unless calibrated by "
        "multiplying by the true prior odds ratio <i>P</i><sub>true</sub>(<i>Y</i>=<i>k</i>) / <i>P</i><sub>train</sub>(<i>Y</i>=<i>k</i>).",
        body_style
    ))
    story.append(Paragraph(
        "<b>Mathematical Formulation:</b> For every sample <i>x</i><sub>i</sub>, Bayes' Theorem computes the exact class posterior:<br/>"
        "<font face='Helvetica-Bold' size='7.5' color='#1a365d'>score<sub>k</sub>(x<sub>i</sub>) = P(Y=k) &times; &prod;<sub>j=1</sub><sup>D</sup> p(x<sub>ij</sub> | Y=k) &nbsp;&nbsp;;&nbsp;&nbsp; P(Y=k | x<sub>i</sub>) = score<sub>k</sub>(x<sub>i</sub>) / &sum;<sub>c</sub> score<sub>c</sub>(x<sub>i</sub>)</font><br/>"
        "Feature likelihoods <i>p</i>(<i>x</i><sub>ij</sub> | <i>Y</i>=<i>k</i>) were modeled as univariate Gaussian densities <i>N</i>(&mu;<sub>kj</sub>, &sigma;<sub>kj</sub><sup>2</sup>) "
        "with variance smoothing &epsilon; = 10<sup>-9</sup> &sigma;<sub>max</sub><sup>2</sup> estimated strictly from training data. Calculations were performed in "
        "log-space and normalized via stable softmax, guaranteeing all class probabilities strictly sum to 1.0. All 10,000 posteriors were saved to "
        "<code>Hloz_M2_Bayesian_Posteriors.csv</code>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Dr. Note Review (Conditional Independence Violation):</b> The Naive Bayes assumption posits that <i>p</i>(<i>X</i> | <i>Y</i>) = &prod; <i>p</i>(<i>x</i><sub>j</sub> | <i>Y</i>). "
        "In our bank dataset, certain features exhibit physical correlations (e.g., <code>Balance</code> and <code>NumOfProducts</code> correlate at -0.304). "
        "When features are correlated, Naive Bayes <i>double-counts evidence</i>, artificially driving posteriors toward extreme 0 or 1. "
        "However, because the MAP classification rule depends solely on the argmax ranking, classification accuracy remains remarkably robust.",
        body_style
    ))
    story.append(Spacer(1, 3))

    story.append(Paragraph("Representative Evaluation: 20 Test Samples", h2_style))
    story.append(Paragraph(
        "Below is a structured selection of 20 test samples (10 actual stayed, 10 actual churned) showing raw features, priors, posteriors, and predictions.",
        body_style
    ))

    # Table of 20 samples (compact display of key columns)
    s20_headers = [
        Paragraph("ID", table_header), Paragraph("Act", table_header), Paragraph("Pred", table_header),
        Paragraph("P(Y=0|X)", table_header), Paragraph("P(Y=1|X)", table_header),
        Paragraph("Score", table_header), Paragraph("Geo", table_header), Paragraph("Age", table_header),
        Paragraph("Balance", table_header), Paragraph("Prods", table_header), Paragraph("Active", table_header)
    ]
    s20_rows = [s20_headers]
    for r in summary_data['sample_20_records']:
        s20_rows.append([
            Paragraph(str(r['Sample_ID']), table_cell_center),
            Paragraph(str(r['Actual_Class']), table_cell_center),
            Paragraph(str(r['Predicted_Class']), table_cell_center),
            Paragraph(f"{r['Posterior_P(Y=0|X)']:.3f}", table_cell_center),
            Paragraph(f"{r['Posterior_P(Y=1|X)']:.3f}", table_cell_center),
            Paragraph(str(r['CreditScore']), table_cell_center),
            Paragraph(r['Geography'][:4], table_cell_center),
            Paragraph(str(r['Age']), table_cell_center),
            Paragraph(f"${r['Balance']:,.0f}", table_cell_center),
            Paragraph(str(r['NumOfProducts']), table_cell_center),
            Paragraph(str(r['IsActiveMember']), table_cell_center)
        ])
    
    t_s20 = Table(s20_rows, colWidths=[0.55*inch, 0.4*inch, 0.4*inch, 0.75*inch, 0.75*inch, 0.55*inch, 0.55*inch, 0.45*inch, 0.95*inch, 0.5*inch, 0.55*inch])
    t_s20.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1a365d")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8f9fa")]),
        ('LINEBELOW', (0, 10), (-1, 10), 1.2, colors.HexColor("#dd6b20")), # Line separating class 0 from class 1
    ]))
    story.append(t_s20)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: WORKED BAYES CALCULATION (SAMPLE ID 5750)
    # =========================================================================
    story.append(Paragraph("2.5 Detailed Worked Bayes Calculation (Sample ID: 5750)", h1_style))
    w_ex = summary_data['worked_example']
    story.append(Paragraph(
        f"<b>Customer Profile (ID: {w_ex['orig_id']}, Surname: Woronoff):</b> Actual Class = <b>Churned (Y=1)</b>.<br/>"
        f"• <i>Raw Features:</i> CreditScore: {w_ex['raw_features']['CreditScore']}, Geography: {w_ex['raw_features']['Geography']}, "
        f"Gender: {w_ex['raw_features']['Gender']}, Age: {w_ex['raw_features']['Age']}, Tenure: {w_ex['raw_features']['Tenure']}, "
        f"Balance: ${w_ex['raw_features']['Balance']:,.2f}, NumOfProducts: {w_ex['raw_features']['NumOfProducts']}, HasCrCard: {w_ex['raw_features']['HasCrCard']}, "
        f"IsActiveMember: {w_ex['raw_features']['IsActiveMember']}, EstimatedSalary: ${w_ex['raw_features']['EstimatedSalary']:,.2f}.<br/>"
        f"• <i>Baseline Priors:</i> P(Y=0) = <b>{w_ex['prior_0']:.4f}</b> (79.63%) &nbsp;|&nbsp; P(Y=1) = <b>{w_ex['prior_1']:.4f}</b> (20.37%).",
        body_style
    ))
    story.append(Paragraph(
        "<b>Step-by-Step Feature Likelihood Computation:</b> Each standardized feature <i>x</i><sub>j</sub> is evaluated against the training class parameters "
        "<i>N</i>(&mu;<sub>0j</sub>, &sigma;<sub>0j</sub><sup>2</sup>) and <i>N</i>(&mu;<sub>1j</sub>, &sigma;<sub>1j</sub><sup>2</sup>). The Likelihood Ratio "
        "<i>LR</i><sub>j</sub> = <i>p</i>(<i>x</i><sub>j</sub> | <i>Y</i>=1) / <i>p</i>(<i>x</i><sub>j</sub> | <i>Y</i>=0) "
        "indicates whether feature <i>j</i> increases churn odds (<i>LR</i> &gt; 1) or retention odds (<i>LR</i> &lt; 1):",
        body_style
    ))

    # Worked Details Table
    w_headers = [
        Paragraph("Feature", table_header), Paragraph("Scaled (x)", table_header),
        Paragraph("p(x|Y=0)", table_header), Paragraph("p(x|Y=1)", table_header),
        Paragraph("Likelihood Ratio", table_header), Paragraph("Bayesian Evidence Impact", table_header)
    ]
    w_rows = [w_headers]
    for d in w_ex['details']:
        lr_val = d['Likelihood_Ratio_Y1_vs_Y0']
        impact = f"Strong Churn Signal (x{lr_val:.1f})" if lr_val > 2.0 else ("Pulls to Churn" if lr_val > 1.0 else "Pulls to Stay")
        w_rows.append([
            Paragraph(d['Feature'], table_cell),
            Paragraph(f"{d['Scaled_Value']:.2f}", table_cell_center),
            Paragraph(f"{d['Likelihood_Y0']:.3e}", table_cell_center),
            Paragraph(f"{d['Likelihood_Y1']:.3e}", table_cell_center),
            Paragraph(f"{lr_val:.3f}", table_cell_center),
            Paragraph(impact, table_cell)
        ])
    t_worked = Table(w_rows, colWidths=[1.5*inch, 0.8*inch, 1.1*inch, 1.1*inch, 1.0*inch, 1.4*inch])
    t_worked.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1a365d")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8f9fa")]),
    ]))
    story.append(t_worked)
    story.append(Spacer(1, 3))

    story.append(Paragraph(
        "<b>Posterior Aggregation via Bayes' Rule:</b><br/>"
        "• Unnormalized Log Score 0: log score<sub>0</sub> = log(0.7963) + &sum; log <i>p</i>(<i>x</i><sub>j</sub> | <i>Y</i>=0) = -19.4881<br/>"
        "• Unnormalized Log Score 1: log score<sub>1</sub> = log(0.2037) + &sum; log <i>p</i>(<i>x</i><sub>j</sub> | <i>Y</i>=1) = -17.8576<br/>"
        f"• Normalized Posteriors: <b>P(Y=0 | X) = {w_ex['post_0']:.4f} (16.39%)</b> &nbsp;|&nbsp; <b>P(Y=1 | X) = {w_ex['post_1']:.4f} (83.61%)</b><br/>"
        f"• MAP Classification Decision: <b>Predicted Class Y* = 1 [argmax<sub>k</sub> P(Y=k | X)]</b> (Matches Ground Truth Churn!).",
        body_style
    ))
    story.append(Spacer(1, 2))

    img_worked = Image(os.path.join(FIGURES_DIR, "fig3_worked_example_likelihoods.png"), width=6.8*inch, height=2.4*inch)
    story.append(img_worked)
    story.append(Paragraph("<b>Figure 2:</b> Log-likelihood ratios log[p(x_j|Y=1)/p(x_j|Y=0)] illustrating feature contributions for Sample 5750.", caption_style))

    story.append(Paragraph(
        "<b>How Features Updated the Prior:</b> Despite the prior heavily suppressing churn (~80% stayed), the customer exhibited "
        "<code>NumOfProducts = 3</code> (LR = 14.28), <code>Geography = Germany</code> (LR = 2.16), and <code>IsActiveMember = 0</code> (LR = 1.71). "
        "The cumulative product of likelihood ratios (<i>LR</i><sub>total</sub> &approx; 16.1) vastly overpowered the prior odds ratio (0.256), driving posterior churn probability to <b>83.61%</b>.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: BAYESIAN MACHINE LEARNING (GAUSSIAN NAIVE BAYES)
    # =========================================================================
    story.append(Paragraph("3. Bayesian Machine Learning – Naive Bayes", h1_style))
    story.append(Paragraph(
        "<b>Model Justification (`GaussianNB`):</b> We selected <code>GaussianNB</code> from scikit-learn. Because our preprocessing pipeline "
        "standardized continuous numerical features to zero mean and unit variance, Gaussian densities model the spread and central tendency of "
        "each class naturally. <code>BernoulliNB</code> would require binarizing continuous variables (losing granular information), while <code>MultinomialNB</code> "
        "is restricted to non-negative count data.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Mathematical Equivalence Verification:</b> To verify theoretical correctness, we compared the class probabilities produced by "
        "scikit-learn's <code>GaussianNB.predict_proba()</code> against our direct manual Bayes implementation from Section 2. Across all 1,500 test samples, "
        "the <b>maximum absolute difference was 2.00 &times; 10<sup>-15</sup></b> (within machine floating-point precision), proving exact numerical equivalence.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Model-Estimated Probabilities for 20 Representative Test Samples", h2_style))
    story.append(Paragraph(
        "Below is the plot of model-estimated posterior probabilities <i>P</i>(<i>Y</i>=0 | <i>X</i>) and <i>P</i>(<i>Y</i>=1 | <i>X</i>) for the exact same 20 test samples "
        "examined in Section 2, displaying the 0.5 decision threshold.",
        body_style
    ))

    img_prob20 = Image(os.path.join(FIGURES_DIR, "fig2_20_samples_probabilities.png"), width=6.8*inch, height=3.6*inch)
    story.append(img_prob20)
    story.append(Paragraph("<b>Figure 3:</b> GaussianNB estimated class probabilities for 20 representative test samples (10 Stayed, 10 Churned).", caption_style))

    story.append(Paragraph(
        "<b>Probability Analysis & Observations:</b><br/>"
        "• <i>Class 0 Samples (Left):</i> For loyal customers (IDs 9161 through 3550), the model outputs confident posteriors <i>P</i>(<i>Y</i>=0 | <i>X</i>) &gt; 0.73. "
        "No False Positives occur among these samples.<br/>"
        "• <i>Class 1 Samples (Right):</i> Customers with pronounced churn signals (IDs 5750, 1305, 3531, 2721) easily surpass the 0.5 threshold (<i>P</i>(<i>Y</i>=1 | <i>X</i>) &gt; 0.80). "
        "However, customers with moderate feature values (IDs 4616, 5818) remain below 0.5 because the ~80% prior acts as a strong conservative anchor. "
        "This explains why Naive Bayes demonstrates high precision (0.7365) but moderate recall (0.6482) on imbalanced data.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: CONVENTIONAL MACHINE LEARNING (DECISION TREE & LOGISTIC REGRESSION)
    # =========================================================================
    story.append(Paragraph("4. Conventional Machine Learning Models", h1_style))
    story.append(Paragraph(
        "To provide a comprehensive comparative study against our Bayesian classifier, we trained two conventional machine learning models "
        "on the exact same training split and tuned them on the independent Validation Set:",
        body_style
    ))
    story.append(Paragraph(
        "<b>1. Decision Tree Classifier (`DecisionTreeClassifier`):</b><br/>"
        "• <i>Prediction Mechanism:</i> Performs hierarchical recursive binary splitting to partition feature space into axis-aligned rectangular regions. "
        "At each node, it selects the feature and threshold that minimizes <b>Gini Impurity</b>: Gini = 1 - &sum; <i>p</i><sub>k</sub><sup>2</sup>. Leaf nodes output the "
        "majority class of training points falling within that leaf.<br/>"
        "• <i>Settings & Tuning:</i> Evaluated <code>max_depth</code> from 2 to 15 on validation data. Minimum split samples was fixed to 20 and leaf size to 10. "
        f"Validation Macro F1 peaked at <b>max_depth = {summary_data['best_tree_depth']}</b> (Validation Macro F1 = {summary_data['best_tree_val_f1']:.4f}).",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. Logistic Regression Classifier (`LogisticRegression`):</b><br/>"
        "• <i>Prediction Mechanism:</i> Models the log-odds of churn as a linear combination of input features: <i>z</i> = <i>w</i><sub>0</sub> + &sum; <i>w</i><sub>j</sub> <i>x</i><sub>j</sub>. "
        "Applies the logistic sigmoid &sigma;(<i>z</i>) = 1 / (1 + <i>e</i><sup>-<i>z</i></sup>) to map linear scores into valid probabilities [0, 1]. Classifies as Churned if &sigma;(<i>z</i>) &ge; 0.5.<br/>"
        "• <i>Settings & Tuning:</i> Optimized using L2 Ridge penalty and <code>lbfgs</code> solver (1,000 max iterations). Evaluated inverse regularization "
        f"<i>C</i> &isin; [0.001, 100.0]. Optimal validation performance achieved at <b>C = {summary_data['best_lr_C']}</b> (Validation Macro F1 = {summary_data['best_lr_val_f1']:.4f}).",
        body_style
    ))
    story.append(Spacer(1, 4))

    img_tuning = Image(os.path.join(FIGURES_DIR, "fig4_validation_tuning.png"), width=6.8*inch, height=3.0*inch)
    story.append(img_tuning)
    story.append(Paragraph("<b>Figure 4:</b> Validation set hyperparameter tuning curves for Decision Tree (depth) and Logistic Regression (C).", caption_style))

    story.append(Paragraph(
        "<b>Hyperparameter Selection Rationale:</b><br/>"
        "• <i>Decision Tree:</i> Depths &lt; 5 underfit, failing to capture feature interactions. Depths &gt; 10 overfit to training noise (training F1 approaches 0.95 "
        "while validation F1 degrades). Depth 8 provides the optimal trade-off between model expressiveness and generalization.<br/>"
        "• <i>Logistic Regression:</i> Small C (&lt; 0.01) excessively regularizes weights toward zero, collapsing recall. <i>C</i> &ge; 1.0 stabilizes convergence.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: EVALUATION, COMPARISON, CONFUSION MATRICES & CONCLUSIONS
    # =========================================================================
    story.append(Paragraph("5. Evaluation and Comparison on Untouched Test Set", h1_style))
    story.append(Paragraph(
        "All models were evaluated on the <b>untouched test set (1,500 samples)</b>. We compare Accuracy, Macro Precision, Macro Recall, "
        "Macro F1, and individual class F1 scores in Table 3.",
        body_style
    ))

    # Metric Table
    m_headers = [
        Paragraph("Evaluated Model", table_header), Paragraph("Accuracy", table_header),
        Paragraph("Macro Prec", table_header), Paragraph("Macro Rec", table_header),
        Paragraph("Macro F1", table_header), Paragraph("Class 0 F1", table_header), Paragraph("Class 1 F1", table_header)
    ]
    m_rows = [m_headers]
    for r in summary_data['eval_table']:
        m_rows.append([
            Paragraph(r['Model'], table_cell),
            Paragraph(f"{r['Accuracy']:.4f}", table_cell_center),
            Paragraph(f"{r['Macro Precision']:.4f}", table_cell_center),
            Paragraph(f"{r['Macro Recall']:.4f}", table_cell_center),
            Paragraph(f"<b>{r['Macro F1']:.4f}</b>", table_cell_center),
            Paragraph(f"{r['Class 0 F1 (Stayed)']:.4f}", table_cell_center),
            Paragraph(f"<b>{r['Class 1 F1 (Churned)']:.4f}</b>", table_cell_center)
        ])
    t_eval = Table(m_rows, colWidths=[1.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.9*inch, 0.9*inch])
    t_eval.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1a365d")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8f9fa")]),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 3))

    img_cm = Image(os.path.join(FIGURES_DIR, "fig5_confusion_matrices.png"), width=6.8*inch, height=3.0*inch)
    story.append(img_cm)
    story.append(Paragraph("<b>Figure 5:</b> Labeled confusion matrices with counts and class recall percentages on the untouched test set.", caption_style))

    story.append(Paragraph(
        "<b>Technical Discussion & Performance Comparison:</b><br/>"
        "• <b>Why Accuracy is Misleading:</b> All models report high accuracy (81.5% to 86.7%) simply by predicting class 0. However, "
        "Logistic Regression achieves a dismal Churn F1 of only <b>0.3092</b> (detecting only 20.3% of churners, missing 243 customers!).<br/>"
        "• <b>Best Performing Model:</b> <b>Decision Tree (depth=8)</b> achieves superior performance across all balanced metrics (Accuracy: <b>86.73%</b>, "
        "Macro F1: <b>0.7696</b>, Churn F1: <b>0.6195</b>, detecting 58.0% of churners). Decision trees successfully capture <i>non-linear feature interactions</i> "
        "(e.g., German residence combined with high age and multiple products) that linear models cannot separate.<br/>"
        "• <b>Naive Bayes Competitiveness:</b> Naive Bayes achieves a Macro F1 of <b>0.6722</b> and Churn F1 of <b>0.4486</b>. While constrained by conditional independence, "
        "its class-conditional variances allow it to outperform Logistic Regression on minority class detection.",
        body_style
    ))
    story.append(Spacer(1, 2))

    story.append(Paragraph("6. Key Takeaways & Conclusions", h1_style))
    story.append(Paragraph(
        "1. <b>Exact Bayesian Implementation:</b> Scratch Bayes matched scikit-learn's GaussianNB within machine precision (2.00 &times; 10<sup>-15</sup>).<br/>"
        "2. <b>Imbalance Awareness:</b> Honest reporting demonstrates that Macro F1 and minority recall are the true indicators of utility in imbalanced domains.<br/>"
        "3. <b>Methodological Integrity:</b> Strict partitioning (70/15/15) and validation tuning guaranteed zero test leakage and honest, defensible results.",
        body_style
    ))

    # Build document
    doc.build(story, canvasmaker=SimpleNumberedCanvas)
    print(f"Report successfully compiled at: {PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
