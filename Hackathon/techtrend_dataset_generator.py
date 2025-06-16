# from reportlab.lib import colors
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
# from reportlab.lib.styles import getSampleStyleSheet

# # Dataset
# data = [
#     ["Question", "Context", "Answer"],
#     ["How many days of annual leave are provided?", "Leave Policy: Employees at TechTrend Innovations are entitled to 20 days...", "20 days of annual leave per year."],
#     ["How long is parental leave?", "Leave Policy: Employees at TechTrend Innovations are entitled to 20 days...", "Up to 12 weeks."],
#     ["When must leave requests be submitted?", "Leave Policy: Employees at TechTrend Innovations are entitled to 20 days...", "At least 7 days in advance via the HR portal."],
#     ["When do annual appraisals take place?", "Appraisal Process: Annual appraisals occur in December...", "In December."],
#     ["By when must self-assessment forms be submitted?", "Appraisal Process: Annual appraisals occur in December...", "By November 15."],
#     ["Are performance bonuses available?", "Appraisal Process: Annual appraisals occur in December...", "Yes, based on appraisal scores."],
#     ["Where can I find my payslip?", "Payslip Access: Payslips are available on the HR portal...", "On the HR portal under 'Payroll'."],
#     ["Can I get paper payslips?", "Payslip Access: Payslips are available on the HR portal...", "Yes, employees can opt for paper payslips."],
#     ["How long to resolve payslip discrepancies?", "Payslip Access: Payslips are available on the HR portal...", "Contact HR within 30 days."],
#     ["How do I claim travel expenses?", "Expense Claims: Employees can claim work-related expenses...", "Submit claims via the HR portal with receipts."],
#     ["What is the deadline for expense claims?", "Expense Claims: Employees can claim work-related expenses...", "Within 60 days."],
#     ["How long for expense approval?", "Expense Claims: Employees can claim work-related expenses...", "5-10 business days."],
#     ["How many training hours are offered?", "Training and Development: TechTrend offers 40 hours of professional...", "40 hours annually."],
#     ["How do I enroll in training courses?", "Training and Development: TechTrend offers 40 hours of professional...", "Via the Learning Portal."],
#     ["Can I attend workshops?", "Training and Development: TechTrend offers 40 hours of professional...", "Yes, workshops are available."],
#     ["How many days can I work remotely?", "Remote Work Policy: Employees may work remotely up to 3 days per week...", "Up to 3 days per week."],
#     ["Do remote workers need manager approval?", "Remote Work Policy: Employees may work remotely up to 3 days per week...", "Yes, subject to manager approval."],
#     ["What is required for a remote workspace?", "Remote Work Policy: Employees may work remotely up to 3 days per week...", "A dedicated workspace and virtual check-ins."],
#     ["How long is the onboarding program?", "Onboarding Process: New hires complete a 2-week onboarding program...", "2 weeks."],
#     ["Do new hires get a buddy?", "Onboarding Process: New hires complete a 2-week onboarding program...", "Yes, for the first 30 days."],
#     ["What training is included in onboarding?", "Onboarding Process: New hires complete a 2-week onboarding program...", "System training and team introductions."],
#     ["Is DEI training mandatory?", "Diversity and Inclusion: TechTrend is committed to a diverse workplace...", "Yes, annually."],
#     ["Can I join affinity groups?", "Diversity and Inclusion: TechTrend is committed to a diverse workplace...", "Yes, for underrepresented communities."],
#     ["What is TechTrend’s diversity commitment?", "Diversity and Inclusion: TechTrend is committed to a diverse workplace...", "Committed to a diverse workplace with DEI initiatives."]
# ]

# # Create PDF
# pdf_file = "TechTrend_Dataset.pdf"
# doc = SimpleDocTemplate(pdf_file, pagesize=letter)
# styles = getSampleStyleSheet()
# elements = []

# # Add title
# title = Paragraph("TechTrend Innovations Policy Dataset", styles['Title'])
# elements.append(title)

# # Create table
# table = Table(data)

# # Style the table
# table.setStyle(TableStyle([
#     ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#     ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#     ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
#     ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#     ('FONTSIZE', (0, 0), (-1, -1), 10),
#     ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
#     ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
#     ('GRID', (0, 0), (-1, -1), 1, colors.black),
#     ('WORDWRAP', (0, 0), (-1, -1), True),
# ]))

# # Adjust column widths
# table._argW[0] = 200  # Question column
# table._argW[1] = 250  # Context column
# table._argW[2] = 150  # Answer column

# elements.append(table)

# # Build PDF
# doc.build(elements)
# print(f"PDF file '{pdf_file}' has been created with the dataset.")
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Dataset
data = [
    ["Question", "Context", "Answer"],
    ["How many days of annual leave are provided?", "Leave Policy: Employees at TechTrend Innovations are entitled to 20 days...", "20 days of annual leave per year."],
    ["How long is parental leave?", "Leave Policy: Employees at TechTrend Innovations are entitled to 20 days...", "Up to 12 weeks."],
    ["When must leave requests be submitted?", "Leave Policy: Employees at TechTrend Innovations are entitled to 20 days...", "At least 7 days in advance via the HR portal."],
    ["When do annual appraisals take place?", "Appraisal Process: Annual appraisals occur in December...", "In December."],
    ["By when must self-assessment forms be submitted?", "Appraisal Process: Annual appraisals occur in December...", "By November 15."],
    ["Are performance bonuses available?", "Appraisal Process: Annual appraisals occur in December...", "Yes, based on appraisal scores."],
    ["Where can I find my payslip?", "Payslip Access: Payslips are available on the HR portal...", "On the HR portal under 'Payroll'."],
    ["Can I get paper payslips?", "Payslip Access: Payslips are available on the HR portal...", "Yes, employees can opt for paper payslips."],
    ["How long to resolve payslip discrepancies?", "Payslip Access: Payslips are available on the HR portal...", "Contact HR within 30 days."],
    ["How do I claim travel expenses?", "Expense Claims: Employees can claim work-related expenses...", "Submit claims via the HR portal with receipts."],
    ["What is the deadline for expense claims?", "Expense Claims: Employees can claim work-related expenses...", "Within 60 days."],
    ["How long for expense approval?", "Expense Claims: Employees can claim work-related expenses...", "5-10 business days."],
    ["How many training hours are offered?", "Training and Development: TechTrend offers 40 hours of professional...", "40 hours annually."],
    ["How do I enroll in training courses?", "Training and Development: TechTrend offers 40 hours of professional...", "Via the Learning Portal."],
    ["Can I attend workshops?", "Training and Development: TechTrend offers 40 hours of professional...", "Yes, workshops are available."],
    ["How many days can I work remotely?", "Remote Work Policy: Employees may work remotely up to 3 days per week...", "Up to 3 days per week."],
    ["Do remote workers need manager approval?", "Remote Work Policy: Employees may work remotely up to 3 days per week...", "Yes, subject to manager approval."],
    ["What is required for a remote workspace?", "Remote Work Policy: Employees may work remotely up to 3 days per week...", "A dedicated workspace and virtual check-ins."],
    ["How long is the onboarding program?", "Onboarding Process: New hires complete a 2-week onboarding program...", "2 weeks."],
    ["Do new hires get a buddy?", "Onboarding Process: New hires complete a 2-week onboarding program...", "Yes, for the first 30 days."],
    ["What training is included in onboarding?", "Onboarding Process: New hires complete a 2-week onboarding program...", "System training and team introductions."],
    ["Is DEI training mandatory?", "Diversity and Inclusion: TechTrend is committed to a diverse workplace...", "Yes, annually."],
    ["Can I join affinity groups?", "Diversity and Inclusion: TechTrend is committed to a diverse workplace...", "Yes, for underrepresented communities."],
    ["What is TechTrend’s diversity commitment?", "Diversity and Inclusion: TechTrend is committed to a diverse workplace...", "Committed to a diverse workplace with DEI initiatives."]
]

# Create PDF
pdf_file = "D:/ihub_task_agentic_ai/Agentic_AI_Workshop/Hackathon/TechTrend_Dataset.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=letter)
styles = getSampleStyleSheet()
elements = []

# Add title
title = Paragraph("TechTrend Innovations Policy Dataset", styles['Title'])
elements.append(title)

# Create table
table = Table(data)

# Simplified table style for better text extraction
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),  # Use a standard font
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.white),  # Simpler background
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    # Removed WORDWRAP to prevent text splitting
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('LEFTPADDING', (0, 0), (-1, -1), 5),
    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
]))

# Adjust column widths
table._argW[0] = 200  # Question column
table._argW[1] = 250  # Context column
table._argW[2] = 150  # Answer column

elements.append(table)

# Build PDF
doc.build(elements)
print(f"PDF file '{pdf_file}' has been created with the dataset.")