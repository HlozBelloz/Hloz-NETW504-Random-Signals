"""
NETW504 - Random Signals and Noise
Milestone 1 - PDF Report Generator
Student: Mazen Mohamed Hamdy Altelbany (ID: 64-12371)
Team Name: Hloz

Generates Hloz_M1_Report.pdf:
- Simple, clear, student-level language
- Exactly 6 pages
- Includes all required tables and plots
"""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

base_dir = r"e:\University\random signals\milestone 1"
figures_dir = os.path.join(base_dir, "figures")
pdf_path = os.path.join(base_dir, "Hloz_M1_Report.pdf")

# Custom Canvas for page numbers and running header
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
        
        # Header on pages 2-6
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "NETW504 Random Signals & Noise — Milestone 1 EDA Report | Team Hloz")
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
        
        # Footer on all pages
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 30, page_text)
        self.drawString(54, 30, "NETW504 — Fall 2026")
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(54, 42, letter[0] - 54, 42)
        
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Simple clean typography
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
        spaceAfter=8
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#333333")
    )
    
    h1_style = ParagraphStyle(
        'H1_Simple',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1a365d"),
        spaceBefore=10,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        'Body_Simple',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#222222"),
        spaceAfter=5
    )

    caption_style = ParagraphStyle(
        'Caption_Simple',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#555555"),
        alignment=1,
        spaceAfter=5
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#222222")
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    story = []

    # ================= PAGE 1 =================
    story.append(Paragraph("NETW504: Random Signals and Noise", subtitle_style))
    story.append(Paragraph("Milestone 1 — Exploratory Data Analysis (EDA) Report", title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a365d"), spaceBefore=2, spaceAfter=8))
    
    meta_text = """
    <b>Instructor:</b> Prof. Talal Elshabrawy &nbsp;|&nbsp; 
    <b>Team Name:</b> Hloz &nbsp;|&nbsp; 
    <b>Student:</b> Mazen Mohamed Hamdy Altelbany (ID: 64-12371)<br/>
    <b>Dataset:</b> Bank Customer Churn Prediction (Kaggle) &nbsp;|&nbsp; 
    <b>Deliverables:</b> Hloz_M1.ipynb, Hloz_M1_Cleaned.csv, Hloz_M1_Report.pdf
    """
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Dataset and Target", h1_style))
    story.append(Paragraph(
        "<b>Dataset Purpose and Link:</b> The dataset used in this milestone is the <i>Bank Customer Churn Prediction</i> from Kaggle. "
        "The goal is to understand what customer features are related to customers leaving the bank (churning). "
        "Kaggle link: <font color='#2b6cb0'>https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction</font>",
        body_style
    ))
    story.append(Paragraph(
        "<b>What One Row Represents:</b> Each row in the dataset represents a single bank customer, containing their personal information "
        "(such as country, gender, age), account numbers (credit score, balance, number of products), activity status, and whether they stayed or left.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Number of Rows and Columns:</b> The raw dataset has <b>10,000 rows (samples)</b> and <b>14 columns (features)</b>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Target Column Name and Classes:</b> The target column is <b><code>Exited</code></b>. It tells us whether the customer stayed or churned. "
        "It is a binary classification problem with 2 classes:",
        body_style
    ))
    
    # Target table
    target_data = [
        [Paragraph("Target Class", table_header), Paragraph("Meaning", table_header), Paragraph("Count (Rows)", table_header), Paragraph("Percentage (%)", table_header)],
        [Paragraph("0", table_cell), Paragraph("Customer stayed (Retained)", table_cell), Paragraph("7,963", table_cell), Paragraph("79.63%", table_cell)],
        [Paragraph("1", table_cell), Paragraph("Customer left (Exited / Churned)", table_cell), Paragraph("2,037", table_cell), Paragraph("20.37%", table_cell)],
        [Paragraph("<b>Total</b>", table_cell), Paragraph("<b>All Customers</b>", table_cell), Paragraph("<b>10,000</b>", table_cell), Paragraph("<b>100.00%</b>", table_cell)]
    ]
    t_target = Table(target_data, colWidths=[1.1*inch, 2.5*inch, 1.5*inch, 1.4*inch])
    t_target.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1a365d")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.HexColor("#ffffff"), colors.HexColor("#f8f9fa")]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#e9ecef")),
    ]))
    story.append(t_target)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Input Features Classification:</b>", h1_style))
    story.append(Paragraph(
        "• <b>ID-like Columns (To Drop):</b> <code>RowNumber</code> (row counter), <code>CustomerId</code> (unique ID), and <code>Surname</code> (customer name). "
        "These are unique identifiers that do not provide general learning signals, so we drop them.<br/>"
        "• <b>Categorical Input Features (2):</b> <code>Geography</code> (France, Spain, Germany) and <code>Gender</code> (Female, Male).<br/>"
        "• <b>Numerical Input Features (8):</b> <code>CreditScore</code>, <code>Age</code>, <code>Tenure</code>, <code>Balance</code>, <code>NumOfProducts</code>, "
        "<code>HasCrCard</code>, <code>IsActiveMember</code>, and <code>EstimatedSalary</code>.",
        body_style
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("2. Initial Inspection and Data Quality", h1_style))
    story.append(Paragraph(
        "• <b>Duplicate Rows:</b> We checked for duplicate rows using <code>df.duplicated().sum()</code>. There are <b>0 duplicate rows</b> in the dataset.<br/>"
        "• <b>Missing Values:</b> We counted missing values across all columns. There are <b>0 missing values</b> in the entire dataset (100% complete).<br/>"
        "• <b>Class Balance:</b> The target is moderately imbalanced (roughly 80% stayed vs 20% left). In future milestones, we should check precision, recall, and F1-score rather than just accuracy.<br/>"
        "• <b>Unusual Values:</b> Over 36% of bank customers have an account balance of exactly 0. This creates a huge spike at 0 in the balance distribution.",
        body_style
    ))
    
    # End Page 1
    story.append(PageBreak())

    # ================= PAGE 2 =================
    story.append(Paragraph("3. Descriptive Statistics for Numerical Features", h1_style))
    story.append(Paragraph(
        "Here is the summary table for the 8 numerical input features ($N = 10,000$), showing count, mean, standard deviation, minimum, median, and maximum.",
        body_style
    ))
    
    stats_data = [
        [Paragraph("Feature", table_header), Paragraph("Count", table_header), Paragraph("Mean", table_header), Paragraph("Std Dev", table_header), Paragraph("Min", table_header), Paragraph("Median", table_header), Paragraph("Max", table_header), Paragraph("Shape", table_header)],
        [Paragraph("<b>CreditScore</b>", table_cell), Paragraph("10,000", table_cell), Paragraph("650.53", table_cell), Paragraph("96.65", table_cell), Paragraph("350.00", table_cell), Paragraph("652.00", table_cell), Paragraph("850.00", table_cell), Paragraph("Bell-shaped", table_cell)],
        [Paragraph("<b>Age</b>", table_cell), Paragraph("10,000", table_cell), Paragraph("38.92", table_cell), Paragraph("10.49", table_cell), Paragraph("18.00", table_cell), Paragraph("37.00", table_cell), Paragraph("92.00", table_cell), Paragraph("Right-skewed", table_cell)],
        [Paragraph("<b>Tenure</b>", table_cell), Paragraph("10,000", table_cell), Paragraph("5.01", table_cell), Paragraph("2.89", table_cell), Paragraph("0.00", table_cell), Paragraph("5.00", table_cell), Paragraph("10.00", table_cell), Paragraph("Uniform (0-10)", table_cell)],
        [Paragraph("<b>Balance</b>", table_cell), Paragraph("10,000", table_cell), Paragraph("76,485.89", table_cell), Paragraph("62,397.41", table_cell), Paragraph("0.00", table_cell), Paragraph("97,198.54", table_cell), Paragraph("250,898.09", table_cell), Paragraph("Bimodal (many 0s)", table_cell)],
        [Paragraph("<b>NumOfProducts</b>", table_cell), Paragraph("10,000", table_cell), Paragraph("1.53", table_cell), Paragraph("0.58", table_cell), Paragraph("1.00", table_cell), Paragraph("1.00", table_cell), Paragraph("4.00", table_cell), Paragraph("Discrete (1-4)", table_cell)],
        [Paragraph("<b>HasCrCard</b>", table_cell), Paragraph("10,000", table_cell), Paragraph("0.71", table_cell), Paragraph("0.46", table_cell), Paragraph("0.00", table_cell), Paragraph("1.00", table_cell), Paragraph("1.00", table_cell), Paragraph("Binary (70.6% 1)", table_cell)],
        [Paragraph("<b>IsActiveMember</b>", table_cell), Paragraph("10,000", table_cell), Paragraph("0.52", table_cell), Paragraph("0.50", table_cell), Paragraph("0.00", table_cell), Paragraph("1.00", table_cell), Paragraph("1.00", table_cell), Paragraph("Binary (51.5% 1)", table_cell)],
        [Paragraph("<b>EstimatedSalary</b>", table_cell), Paragraph("10,000", table_cell), Paragraph("100,090.24", table_cell), Paragraph("57,510.50", table_cell), Paragraph("11.58", table_cell), Paragraph("100,193.92", table_cell), Paragraph("199,992.48", table_cell), Paragraph("Uniform", table_cell)],
    ]
    t_stats = Table(stats_data, colWidths=[1.1*inch, 0.65*inch, 0.85*inch, 0.85*inch, 0.65*inch, 0.85*inch, 0.85*inch, 1.4*inch])
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1a365d")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8f9fa")]),
    ]))
    story.append(t_stats)
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. Target Distribution and Missing Values Plots", h1_style))
    story.append(Paragraph(
        "Below are the plots showing the target class distribution and the missing values check before preprocessing.",
        body_style
    ))
    
    img_target = Image(os.path.join(figures_dir, "target_distribution.png"), width=3.3*inch, height=2.2*inch)
    img_missing = Image(os.path.join(figures_dir, "missing_values.png"), width=3.3*inch, height=2.2*inch)
    t_fig_row1 = Table([[img_target, img_missing]], colWidths=[3.4*inch, 3.4*inch])
    t_fig_row1.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(t_fig_row1)
    
    cap_row1 = Table([
        [Paragraph("<b>Figure 1:</b> Number of samples in each target class (<code>Exited</code>).", caption_style),
         Paragraph("<b>Figure 2:</b> Missing values count for all columns before cleaning.", caption_style)]
    ], colWidths=[3.4*inch, 3.4*inch])
    story.append(cap_row1)
    
    story.append(Paragraph(
        "<b>Observations:</b><br/>"
        "• <b>Target Distribution:</b> 7,963 customers (79.6%) stayed, and 2,037 customers (20.4%) left. The target is imbalanced with a 4:1 ratio.<br/>"
        "• <b>Missing Values:</b> Every single column has 0 missing values. The dataset is fully complete.",
        body_style
    ))

    # End Page 2
    story.append(PageBreak())

    # ================= PAGE 3 =================
    story.append(Paragraph("5. Estimated PDF for Numerical Features", h1_style))
    story.append(Paragraph(
        "Here we plot the estimated Probability Density Function (PDF) using normalized histograms (<code>density=True</code>) with a KDE curve for all 8 numerical features.",
        body_style
    ))
    
    img_pdfs = Image(os.path.join(figures_dir, "numerical_pdfs.png"), width=6.6*inch, height=7.2*inch)
    story.append(img_pdfs)
    story.append(Paragraph("<b>Figure 3:</b> Estimated PDF plots with KDE curves for all 8 numerical features.", caption_style))
    
    story.append(Paragraph(
        "<b>Observation:</b> <code>CreditScore</code> has a symmetric, bell-shaped distribution centered around 650. <code>Age</code> is skewed to the right, with most customers between 30 and 45 years old. "
        "<code>Balance</code> has a large spike at zero (customers with zero balance) and another bell shape around 120,000 for customers with money in the bank. "
        "<code>EstimatedSalary</code> is flat across the whole range from 0 to 200,000, looking like a uniform distribution.",
        body_style
    ))

    # End Page 3
    story.append(PageBreak())

    # ================= PAGE 4 =================
    story.append(Paragraph("6. Empirical CDF for Numerical Features", h1_style))
    story.append(Paragraph(
        "Below are the Empirical Cumulative Distribution Function (CDF) step plots for all 8 numerical features, showing cumulative probability $F(x)$.",
        body_style
    ))
    
    img_cdfs = Image(os.path.join(figures_dir, "numerical_cdfs.png"), width=6.6*inch, height=7.2*inch)
    story.append(img_cdfs)
    story.append(Paragraph("<b>Figure 4:</b> Empirical CDF step plots for all 8 numerical features.", caption_style))
    
    story.append(Paragraph(
        "<b>Observation:</b> The CDF for <code>Balance</code> jumps vertically at 0 up to about 0.36, which confirms that roughly 36% of customers have zero balance. "
        "The CDF of <code>EstimatedSalary</code> is a straight diagonal line from (0, 0) to (200000, 1), which is the expected shape of a uniform distribution. "
        "The CDFs of <code>CreditScore</code> and <code>Age</code> have smooth S-curves.",
        body_style
    ))

    # End Page 4
    story.append(PageBreak())

    # ================= PAGE 5 =================
    story.append(Paragraph("7. Correlation Matrix and Heatmap", h1_style))
    story.append(Paragraph(
        "Figure 5 shows the correlation heatmap between numerical features after preprocessing. Figure 6 shows boxplots comparing Age and CreditScore between customers who stayed vs left.",
        body_style
    ))
    
    img_corr = Image(os.path.join(figures_dir, "correlation_heatmap.png"), width=4.1*inch, height=3.3*inch)
    img_box = Image(os.path.join(figures_dir, "boxplots_key_features.png"), width=2.7*inch, height=3.3*inch)
    
    t_fig_row2 = Table([[img_corr, img_box]], colWidths=[4.1*inch, 2.7*inch])
    t_fig_row2.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(t_fig_row2)
    
    cap_row2 = Table([
        [Paragraph("<b>Figure 5:</b> Correlation heatmap of numerical features.", caption_style),
         Paragraph("<b>Figure 6:</b> Boxplots of Age and CreditScore vs Exited.", caption_style)]
    ], colWidths=[4.1*inch, 2.7*inch])
    story.append(cap_row2)

    story.append(Paragraph(
        "<b>Observation:</b> Almost all numerical features have correlations close to 0. "
        "The only small correlation is between <code>Balance</code> and <code>NumOfProducts</code> (-0.30), meaning customers with more products tend to have lower account balances. "
        "From Figure 6, older customers are much more likely to exit (churned customers have a higher median age of ~45 compared to ~36 for retained customers). "
        "On the other hand, credit scores are almost identical between both groups.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("8. Basic Preprocessing Summary", h1_style))
    story.append(Paragraph(
        "We followed the 4 required preprocessing steps in strict order:",
        body_style
    ))
    story.append(Paragraph(
        "1. <b>Remove duplicate rows:</b> 0 rows removed (no duplicate rows existed).<br/>"
        "2. <b>Remove rows with missing target:</b> 0 rows removed (all rows have a target label).<br/>"
        "3. <b>Fill missing numerical values with median:</b> 0 values filled (no null values were present).<br/>"
        "4. <b>Encode categorical features and target:</b> Dropped ID columns (<code>RowNumber</code>, <code>CustomerId</code>, <code>Surname</code>). "
        "One-hot encoded <code>Geography</code> into 3 binary columns (<code>Geography_France</code>, <code>Geography_Germany</code>, <code>Geography_Spain</code>) "
        "and <code>Gender</code> into 2 binary columns (<code>Gender_Female</code>, <code>Gender_Male</code>). Target <code>Exited</code> was kept as integer labels (0 and 1).",
        body_style
    ))

    # End Page 5
    story.append(PageBreak())

    # ================= PAGE 6 =================
    story.append(Paragraph("9. Final Cleaned Dataset Specification", h1_style))
    story.append(Paragraph(
        "The cleaned dataset was saved to <b><code>Hloz_M1_Cleaned.csv</code></b>. It has <b>10,000 rows</b> and <b>14 columns</b>. Here is the column dictionary:",
        body_style
    ))
    
    clean_dict_data = [
        [Paragraph("Column Name", table_header), Paragraph("Data Type", table_header), Paragraph("Transformation", table_header), Paragraph("Values / Range", table_header)],
        [Paragraph("<code>CreditScore</code>", table_cell), Paragraph("Integer", table_cell), Paragraph("Raw numerical", table_cell), Paragraph("[350, 850]", table_cell)],
        [Paragraph("<code>Age</code>", table_cell), Paragraph("Integer", table_cell), Paragraph("Raw numerical", table_cell), Paragraph("[18, 92]", table_cell)],
        [Paragraph("<code>Tenure</code>", table_cell), Paragraph("Integer", table_cell), Paragraph("Raw numerical", table_cell), Paragraph("[0, 10]", table_cell)],
        [Paragraph("<code>Balance</code>", table_cell), Paragraph("Float", table_cell), Paragraph("Raw numerical", table_cell), Paragraph("[0.00, 250,898.09]", table_cell)],
        [Paragraph("<code>NumOfProducts</code>", table_cell), Paragraph("Integer", table_cell), Paragraph("Raw numerical", table_cell), Paragraph("[1, 4]", table_cell)],
        [Paragraph("<code>HasCrCard</code>", table_cell), Paragraph("Integer", table_cell), Paragraph("Binary indicator", table_cell), Paragraph("{0, 1}", table_cell)],
        [Paragraph("<code>IsActiveMember</code>", table_cell), Paragraph("Integer", table_cell), Paragraph("Binary indicator", table_cell), Paragraph("{0, 1}", table_cell)],
        [Paragraph("<code>EstimatedSalary</code>", table_cell), Paragraph("Float", table_cell), Paragraph("Raw numerical", table_cell), Paragraph("[11.58, 199,992.48]", table_cell)],
        [Paragraph("<code>Exited</code>", table_cell), Paragraph("Integer", table_cell), Paragraph("Target (0: Stayed, 1: Left)", table_cell), Paragraph("{0, 1}", table_cell)],
        [Paragraph("<code>Geography_France</code>", table_cell), Paragraph("Integer", table_cell), Paragraph("One-Hot Encoded from Geography", table_cell), Paragraph("{0, 1}", table_cell)],
        [Paragraph("<code>Geography_Germany</code>", table_cell), Paragraph("Integer", table_cell), Paragraph("One-Hot Encoded from Geography", table_cell), Paragraph("{0, 1}", table_cell)],
        [Paragraph("<code>Geography_Spain</code>", table_cell), Paragraph("Integer", table_cell), Paragraph("One-Hot Encoded from Geography", table_cell), Paragraph("{0, 1}", table_cell)],
        [Paragraph("<code>Gender_Female</code>", table_cell), Paragraph("Integer", table_cell), Paragraph("One-Hot Encoded from Gender", table_cell), Paragraph("{0, 1}", table_cell)],
        [Paragraph("<code>Gender_Male</code>", table_cell), Paragraph("Integer", table_cell), Paragraph("One-Hot Encoded from Gender", table_cell), Paragraph("{0, 1}", table_cell)],
    ]
    t_clean = Table(clean_dict_data, colWidths=[1.5*inch, 0.8*inch, 2.5*inch, 1.8*inch])
    t_clean.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1a365d")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8f9fa")]),
    ]))
    story.append(t_clean)
    story.append(Spacer(1, 10))

    story.append(Paragraph("10. Conclusion and What We Learned (5–8 Lines)", h1_style))
    conclusion_text = (
        "In this milestone, we performed an exploratory data analysis on the bank churn dataset with 10,000 customers. "
        "The data is very clean with zero duplicate rows and zero missing values across all features. "
        "We observed that the target variable <code>Exited</code> is imbalanced, with roughly 79.6% of customers staying and 20.4% leaving. "
        "Distribution plots showed that <code>CreditScore</code> is normal, <code>EstimatedSalary</code> is uniform, and <code>Balance</code> has a large group of customers with zero balance. "
        "Our analysis showed that age is strongly related to churn, with older customers leaving more frequently than younger ones. "
        "Finally, we removed useless ID columns and one-hot encoded the categorical features, saving a clean dataset of 10,000 rows and 14 columns into <code>Hloz_M1_Cleaned.csv</code>."
    )
    story.append(Paragraph(conclusion_text, body_style))
    story.append(Spacer(1, 10))

    # Deliverables verification box
    box_data = [
        [Paragraph("<b>Milestone 1 Deliverables Summary</b>", table_header)],
        [Paragraph(
            "✓ <b>Hloz_M1_Report.pdf:</b> Summary report within the 6-page limit.<br/>"
            "✓ <b>Hloz_M1.ipynb:</b> Executable Jupyter notebook with code, outputs, and written observations.<br/>"
            "✓ <b>Hloz_M1_Cleaned.csv:</b> Final cleaned and encoded dataset ready for Milestone 2.",
            table_cell
        )]
    ]
    t_box = Table(box_data, colWidths=[6.6*inch])
    t_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#f0f4f8")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#2b6cb0")),
    ]))
    story.append(t_box)

    doc.build(story, canvasmaker=SimpleNumberedCanvas)
    print(f"Report built successfully at: {pdf_path}")

if __name__ == "__main__":
    build_pdf()
