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
    story.append(Paragraph("Milestone 1 — Exploratory Data Analysis Report", title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a365d"), spaceBefore=2, spaceAfter=8))
    
    meta_text = """
    <b>Instructor:</b> Prof. Talal Elshabrawy &nbsp;|&nbsp; 
    <b>Team Name:</b> Hloz<br/>
    <b>Team Members:</b> Mazen Mohamed Hamdy Altelbany (64-12371), Malak Sherif Mohamed (64-10784), Suhad Eyhab Rasheed (64-31506), Sama Ismael Ahel (64-19880)<br/>
    <b>Dataset:</b> Bank Customer Churn Prediction (Kaggle)
    """
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Dataset and Target", h1_style))
    story.append(Paragraph(
        "<b>Dataset and Source:</b> We chose the <i>Bank Customer Churn Prediction</i> dataset from Kaggle. "
        "The goal is to study customer attributes and see which ones relate to whether a customer leaves the bank. "
        "Link: <font color='#2b6cb0'>https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction</font>",
        body_style
    ))
    story.append(Paragraph(
        "<b>Row Representation:</b> Each row in the table represents a single bank customer. It includes their basic background "
        "(country, gender, age), account information (credit score, balance, number of products held), activity status, and whether they stayed or exited.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Dataset Size:</b> The raw dataset contains <b>10,000 rows</b> and <b>14 columns</b>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Target Column:</b> The target column is <b><code>Exited</code></b>, which marks whether a customer churned or not. "
        "This is a binary classification target with 2 classes:",
        body_style
    ))
    
    # Target table
    target_data = [
        [Paragraph("Target Class", table_header), Paragraph("Meaning", table_header), Paragraph("Count", table_header), Paragraph("Percentage", table_header)],
        [Paragraph("0", table_cell), Paragraph("Customer stayed (retained)", table_cell), Paragraph("7,963", table_cell), Paragraph("79.63%", table_cell)],
        [Paragraph("1", table_cell), Paragraph("Customer left (exited)", table_cell), Paragraph("2,037", table_cell), Paragraph("20.37%", table_cell)],
        [Paragraph("<b>Total</b>", table_cell), Paragraph("<b>All rows</b>", table_cell), Paragraph("<b>10,000</b>", table_cell), Paragraph("<b>100.00%</b>", table_cell)]
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

    story.append(Paragraph("<b>Input Features:</b>", h1_style))
    story.append(Paragraph(
        "• <b>Identifier Columns (Dropped):</b> <code>RowNumber</code>, <code>CustomerId</code>, and <code>Surname</code>. "
        "These are IDs and customer names that don't help in predicting churn, so we dropped them.<br/>"
        "• <b>Categorical Features (2):</b> <code>Geography</code> (France, Spain, Germany) and <code>Gender</code> (Female, Male).<br/>"
        "• <b>Numerical Features (8):</b> <code>CreditScore</code>, <code>Age</code>, <code>Tenure</code>, <code>Balance</code>, <code>NumOfProducts</code>, "
        "<code>HasCrCard</code>, <code>IsActiveMember</code>, and <code>EstimatedSalary</code>.",
        body_style
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("2. Initial Inspection and Data Quality", h1_style))
    story.append(Paragraph(
        "• <b>Duplicates:</b> We ran <code>df.duplicated().sum()</code> and found <b>0 duplicate rows</b>.<br/>"
        "• <b>Missing Values:</b> We checked <code>df.isnull().sum()</code> across all columns and found <b>0 missing values</b> (the dataset is 100% complete).<br/>"
        "• <b>Class Imbalance:</b> Around 80% stayed and 20% left. Because of this imbalance, accuracy alone won't be enough later on, and we will need precision and recall.<br/>"
        "• <b>Unusual Observations:</b> Exactly 3,617 customers (about 36.2%) have a balance of 0, creating a sharp spike at zero in the balance distribution.",
        body_style
    ))
    
    # End Page 1
    story.append(PageBreak())

    # ================= PAGE 2 =================
    story.append(Paragraph("3. Descriptive Statistics for Numerical Features", h1_style))
    story.append(Paragraph(
        "Summary table for the 8 numerical features ($N = 10,000$), listing count, mean, standard deviation, min, median, and max.",
        body_style
    ))
    
    stats_data = [
        [Paragraph("Feature", table_header), Paragraph("Count", table_header), Paragraph("Mean", table_header), Paragraph("Std Dev", table_header), Paragraph("Min", table_header), Paragraph("Median", table_header), Paragraph("Max", table_header), Paragraph("Distribution", table_header)],
        [Paragraph("<b>CreditScore</b>", table_cell), Paragraph("10,000", table_cell), Paragraph("650.53", table_cell), Paragraph("96.65", table_cell), Paragraph("350.00", table_cell), Paragraph("652.00", table_cell), Paragraph("850.00", table_cell), Paragraph("Bell-shaped", table_cell)],
        [Paragraph("<b>Age</b>", table_cell), Paragraph("10,000", table_cell), Paragraph("38.92", table_cell), Paragraph("10.49", table_cell), Paragraph("18.00", table_cell), Paragraph("37.00", table_cell), Paragraph("92.00", table_cell), Paragraph("Right-skewed", table_cell)],
        [Paragraph("<b>Tenure</b>", table_cell), Paragraph("10,000", table_cell), Paragraph("5.01", table_cell), Paragraph("2.89", table_cell), Paragraph("0.00", table_cell), Paragraph("5.00", table_cell), Paragraph("10.00", table_cell), Paragraph("Uniform (0-10)", table_cell)],
        [Paragraph("<b>Balance</b>", table_cell), Paragraph("10,000", table_cell), Paragraph("76,485.89", table_cell), Paragraph("62,397.41", table_cell), Paragraph("0.00", table_cell), Paragraph("97,198.54", table_cell), Paragraph("250,898.09", table_cell), Paragraph("Many zeros", table_cell)],
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
        "Here are the bar plots for target class balance and missing values check before any preprocessing.",
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
        [Paragraph("<b>Figure 1:</b> Counts for target column <code>Exited</code>.", caption_style),
         Paragraph("<b>Figure 2:</b> Missing values count per column.", caption_style)]
    ], colWidths=[3.4*inch, 3.4*inch])
    story.append(cap_row1)
    
    story.append(Paragraph(
        "<b>Observations:</b><br/>"
        "• <b>Target:</b> 7,963 customers stayed (79.6%) and 2,037 left (20.4%). The classes are imbalanced at roughly 4 to 1.<br/>"
        "• <b>Missing Values:</b> None of the columns have missing entries (all counts are 0).",
        body_style
    ))

    # End Page 2
    story.append(PageBreak())

    # ================= PAGE 3 =================
    story.append(Paragraph("5. Estimated PDF for Numerical Features", h1_style))
    story.append(Paragraph(
        "We plotted normalized histograms (<code>density=True</code>) along with KDE curves for each of the 8 numerical features to see their shapes.",
        body_style
    ))
    
    img_pdfs = Image(os.path.join(figures_dir, "numerical_pdfs.png"), width=6.6*inch, height=7.2*inch)
    story.append(img_pdfs)
    story.append(Paragraph("<b>Figure 3:</b> Estimated PDF histograms with KDE curves for numerical features.", caption_style))
    
    story.append(Paragraph(
        "<b>Observations:</b> <code>CreditScore</code> looks like a normal curve centered around 650. <code>Age</code> is skewed to the right, showing that most customers are in their 30s and 40s. "
        "<code>Balance</code> has a huge spike at 0 (people with no money in the account) and a second bump near 120,000. "
        "<code>EstimatedSalary</code> is flat from 0 to 200,000, which behaves like a uniform distribution.",
        body_style
    ))

    # End Page 3
    story.append(PageBreak())

    # ================= PAGE 4 =================
    story.append(Paragraph("6. Empirical CDF for Numerical Features", h1_style))
    story.append(Paragraph(
        "We plotted the empirical CDF step curves to inspect cumulative probabilities for each feature.",
        body_style
    ))
    
    img_cdfs = Image(os.path.join(figures_dir, "numerical_cdfs.png"), width=6.6*inch, height=7.2*inch)
    story.append(img_cdfs)
    story.append(Paragraph("<b>Figure 4:</b> Empirical CDF step plots for numerical features.", caption_style))
    
    story.append(Paragraph(
        "<b>Observations:</b> The CDF for <code>Balance</code> jumps straight up at zero to about 0.36, confirming that about 36% of accounts have zero balance. "
        "<code>EstimatedSalary</code> shows an almost straight 45-degree diagonal line, which matches a uniform distribution. "
        "<code>CreditScore</code> and <code>Age</code> follow smooth S-shaped curves.",
        body_style
    ))

    # End Page 4
    story.append(PageBreak())

    # ================= PAGE 5 =================
    story.append(Paragraph("7. Correlation Matrix and Heatmap", h1_style))
    story.append(Paragraph(
        "Figure 5 shows the correlation matrix between numerical features. Figure 6 shows boxplots comparing Age and CreditScore for customers who stayed vs left.",
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
         Paragraph("<b>Figure 6:</b> Boxplots of Age and CreditScore grouped by Exited.", caption_style)]
    ], colWidths=[4.1*inch, 2.7*inch])
    story.append(cap_row2)
    
    story.append(Paragraph(
        "<b>Observations:</b> Most numerical features have very low correlations with each other. "
        "The strongest correlation is -0.30 between <code>Balance</code> and <code>NumOfProducts</code>, meaning customers with more products tend to keep lower balances. "
        "In Figure 6, churned customers have a higher median age (~45) than retained customers (~36). Meanwhile, credit score values look practically the same for both groups.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("8. Basic Preprocessing Summary", h1_style))
    story.append(Paragraph(
        "We followed the 4 required steps in order:",
        body_style
    ))
    story.append(Paragraph(
        "1. <b>Duplicate rows:</b> Checked with <code>drop_duplicates()</code>; 0 rows removed.<br/>"
        "2. <b>Missing targets:</b> Checked with <code>dropna(subset=['Exited'])</code>; 0 rows removed.<br/>"
        "3. <b>Missing numericals:</b> Checked for null values; 0 imputed because the columns had no missing entries.<br/>"
        "4. <b>Encoding:</b> Dropped non-predictive IDs (<code>RowNumber</code>, <code>CustomerId</code>, <code>Surname</code>). "
        "One-hot encoded <code>Geography</code> into 3 binary columns (France, Germany, Spain) "
        "and <code>Gender</code> into 2 columns (Female, Male). Kept target <code>Exited</code> as integer labels (0 and 1).",
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

    story.append(Paragraph("10. Conclusion and What We Learned", h1_style))
    conclusion_text = (
        "In this milestone, we explored the bank churn dataset of 10,000 customers. "
        "The data is very clean with no duplicate rows and no missing values in any feature. "
        "We found that the target variable <code>Exited</code> is imbalanced, where around 80% of customers stayed and 20% left. "
        "Looking at the distributions, <code>CreditScore</code> is approximately bell-shaped, <code>EstimatedSalary</code> is evenly spread across its range, and <code>Balance</code> has a noticeable peak at zero from customers with empty balances. "
        "From our correlation and boxplots, customer age has the clearest relationship with churn, as older customers tend to leave more often than younger ones. "
        "Finally, we removed unnecessary identification columns and one-hot encoded the categorical variables, leaving us with a clean table of 10,000 rows and 14 columns ready for modeling."
    )
    story.append(Paragraph(conclusion_text, body_style))

    doc.build(story, canvasmaker=SimpleNumberedCanvas)
    print(f"Report built successfully at: {pdf_path}")

if __name__ == "__main__":
    build_pdf()
