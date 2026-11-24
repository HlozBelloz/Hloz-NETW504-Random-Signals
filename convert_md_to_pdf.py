"""
Generate PDF versions of:
1. EXPLANATION_AND_WALKTHROUGH.md -> EXPLANATION_AND_WALKTHROUGH.pdf
2. milestone 1/README.md -> milestone 1/README.pdf

Uses ReportLab to produce clean, well-formatted, professional PDFs.
"""

import os
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Preformatted
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
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
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "NETW504 Random Signals & Noise — Milestone 1 Study Reference")
            self.setStrokeColor(colors.HexColor("#dddddd"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
            
        # Footer (all pages)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 30, page_text)
        self.drawString(54, 30, "Team Hloz — NETW504")
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(54, 42, letter[0] - 54, 42)
        self.restoreState()

def markdown_to_pdf(md_file_path, pdf_file_path, doc_title):
    doc = SimpleDocTemplate(
        pdf_file_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=6
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1a365d"),
        spaceBefore=12,
        spaceAfter=4
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2b6cb0"),
        spaceBefore=8,
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#222222"),
        spaceAfter=4
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=15,
        spaceAfter=2
    )
    code_style = ParagraphStyle(
        'CodeText',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1a202c")
    )
    table_cell = ParagraphStyle(
        'TCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#222222")
    )
    table_hdr = ParagraphStyle(
        'THdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    with open(md_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    story = []
    in_code_block = False
    code_buffer = []
    in_table = False
    table_rows = []

    def flush_table():
        nonlocal in_table, table_rows
        if not table_rows:
            in_table = False
            return
        
        # Prepare ReportLab Table
        data = []
        is_first = True
        for row in table_rows:
            parsed_row = []
            for col in row:
                st = table_hdr if is_first else table_cell
                # Clean col formatting
                clean_col = col.replace("`", "<code>").replace("`", "</code>")
                clean_col = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_col)
                parsed_row.append(Paragraph(clean_col, st))
            data.append(parsed_row)
            is_first = False

        if data:
            col_count = len(data[0])
            col_width = (6.6 * inch) / col_count
            t = Table(data, colWidths=[col_width]*col_count)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1a365d")),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8f9fa")]),
            ]))
            story.append(t)
            story.append(Spacer(1, 6))

        table_rows = []
        in_table = False

    for line in lines:
        raw = line.rstrip('\r\n')
        stripped = raw.strip()

        # Handle Code blocks
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "\n".join(code_buffer)
                code_buffer = []
                t_code = Table([[Preformatted(code_text, code_style)]], colWidths=[6.6*inch])
                t_code.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                    ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ]))
                story.append(t_code)
                story.append(Spacer(1, 4))
            else:
                if in_table:
                    flush_table()
                in_code_block = True
                code_buffer = []
            continue

        if in_code_block:
            code_buffer.append(raw)
            continue

        # Handle Tables
        if stripped.startswith("|") and stripped.endswith("|"):
            parts = [p.strip() for p in stripped[1:-1].split("|")]
            # Check if divider line
            if all(re.match(r'^:?-+:?$', p) for p in parts if p):
                continue
            table_rows.append(parts)
            in_table = True
            continue
        elif in_table:
            flush_table()

        if not stripped:
            continue

        # Horizontal Rule
        if stripped in ["---", "***", "___"]:
            story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceBefore=4, spaceAfter=6))
            continue

        # Headers
        if stripped.startswith("# "):
            story.append(Paragraph(stripped[2:], title_style))
            continue
        if stripped.startswith("## "):
            story.append(Paragraph(stripped[3:], h1_style))
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(stripped[4:], h2_style))
            continue

        # Format markdown inline elements (bold, italic, code)
        text = stripped
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
        # Markdown links [text](url) -> text
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)

        # Bullets
        if stripped.startswith("- ") or stripped.startswith("* "):
            story.append(Paragraph(f"&bull; {text[2:]}", bullet_style))
        elif re.match(r'^\d+\.\s', stripped):
            num = stripped.split(".")[0]
            story.append(Paragraph(f"<b>{num}.</b> {text[len(num)+2:]}", bullet_style))
        else:
            story.append(Paragraph(text, body_style))

    if in_table:
        flush_table()

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated: {pdf_file_path}")

if __name__ == "__main__":
    base_dir = r"e:\University\random signals"
    
    # 1. EXPLANATION_AND_WALKTHROUGH.pdf
    md1 = os.path.join(base_dir, "EXPLANATION_AND_WALKTHROUGH.md")
    pdf1 = os.path.join(base_dir, "EXPLANATION_AND_WALKTHROUGH.pdf")
    markdown_to_pdf(md1, pdf1, "Milestone 1 Explanation and Walkthrough")
    
    # 2. milestone 1/README.pdf
    md2 = os.path.join(base_dir, "milestone 1", "README.md")
    pdf2 = os.path.join(base_dir, "milestone 1", "README.pdf")
    markdown_to_pdf(md2, pdf2, "Milestone 1 Project Guide")
