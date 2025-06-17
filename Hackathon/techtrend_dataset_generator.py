from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Expanded Dataset
data = [
    ["Question", "Context", "Answer"],
    ["How many days of annual leave are provided?", "Leave Policy: Employees at TechTrend Innovations are entitled to 20 days of annual leave per calendar year, accruing monthly.", "20 days of annual leave per year."],
    ["How long is parental leave?", "Leave Policy: Parental leave at TechTrend Innovations includes maternity and paternity leave, offering up to 12 weeks for primary caregivers.", "Up to 12 weeks."],
    ["When must leave requests be submitted?", "Leave Policy: All leave requests must be submitted through the HR portal with manager approval.", "At least 7 days in advance via the HR portal."],
    ["What is the process for emergency leave?", "Leave Policy: Emergency leave can be requested with less than 7 days’ notice in exceptional circumstances, subject to HR approval.", "Submit via HR portal; approval within 24 hours."],
    ["Can I carry over unused leave?", "Leave Policy: Up to 5 days of unused annual leave can be carried over to the next year with manager approval.", "Up to 5 days with manager approval."],
    ["When do annual appraisals take place?", "Appraisal Process: Annual appraisals occur in December to review performance and set goals.", "In December."],
    ["By when must self-assessment forms be submitted?", "Appraisal Process: Employees must submit self-assessment forms to prepare for appraisals.", "By November 15."],
    ["Are performance bonuses available?", "Appraisal Process: Performance bonuses are awarded based on appraisal outcomes.", "Yes, based on appraisal scores."],
    ["How is the appraisal score calculated?", "Appraisal Process: Scores are based on KPIs, peer feedback, and manager evaluation.", "Based on KPIs, peer feedback, and manager evaluation."],
    ["Where can I find my payslip?", "Payslip Access: Payslips are available on the HR portal under the Payroll section.", "On the HR portal under 'Payroll'."],
    ["Can I get paper payslips?", "Payslip Access: Employees can opt for paper payslips via HR request.", "Yes, employees can opt for paper payslips."],
    ["How long to resolve payslip discrepancies?", "Payslip Access: Discrepancies must be reported to HR for resolution.", "Contact HR within 30 days."],
    ["How do I claim travel expenses?", "Expense Claims: Work-related travel expenses are reimbursable with valid receipts.", "Submit claims via the HR portal with receipts."],
    ["What is the deadline for expense claims?", "Expense Claims: Claims must be submitted within 60 days of incurring expenses.", "Within 60 days."],
    ["How long for expense approval?", "Expense Claims: Approval process takes 5-10 business days after submission.", "5-10 business days."],
    ["What expenses are eligible for reimbursement?", "Expense Claims: Eligible expenses include travel, accommodation, and work-related supplies.", "Travel, accommodation, and work-related supplies."],
    ["How many training hours are offered?", "Training and Development: TechTrend offers 40 hours of professional development annually.", "40 hours annually."],
    ["How do I enroll in training courses?", "Training and Development: Courses are available via the Learning Portal.", "Via the Learning Portal."],
    ["Can I attend workshops?", "Training and Development: Workshops count toward annual training hours.", "Yes, workshops are available."],
    ["Are external certifications reimbursable?", "Training and Development: Approved external certifications are fully reimbursable.", "Yes, with prior approval."],
    ["How many days can I work remotely?", "Remote Work Policy: Employees may work remotely up to 3 days per week, subject to role requirements.", "Up to 3 days per week."],
    ["Do remote workers need manager approval?", "Remote Work Policy: Remote work arrangements require manager approval.", "Yes, subject to manager approval."],
    ["What is required for a remote workspace?", "Remote Work Policy: Remote employees need a dedicated workspace and stable internet.", "A dedicated workspace and virtual check-ins."],
    ["How long is the onboarding program?", "Onboarding Process: New hires complete a structured onboarding program.", "2 weeks."],
    ["Do new hires get a buddy?", "Onboarding Process: A buddy is assigned to support new hires.", "Yes, for the first 30 days."],
    ["What training is included in onboarding?", "Onboarding Process: Onboarding includes system training and team introductions.", "System training and team introductions."],
    ["Is DEI training mandatory?", "Diversity and Inclusion: TechTrend requires annual DEI training for all employees.", "Yes, annually."],
    ["Can I join affinity groups?", "Diversity and Inclusion: Affinity groups are open to employees from underrepresented communities.", "Yes, for underrepresented communities."],
    ["What is TechTrend’s diversity commitment?", "Diversity and Inclusion: TechTrend is committed to fostering a diverse and inclusive workplace.", "Committed to a diverse workplace with DEI initiatives."],
    ["How do I report a harassment issue?", "Workplace Conduct: Harassment issues must be reported to HR immediately for investigation.", "Report to HR immediately."],
    ["What support is available for mental health?", "Employee Wellness: TechTrend offers an Employee Assistance Program (EAP) for mental health support.", "Employee Assistance Program (EAP)."],
    ["How are workplace disputes resolved?", "Workplace Conduct: Disputes are handled through mediation and HR review.", "Through mediation and HR review."],
    ["What is the process for requesting accommodations?", "Employee Support: Reasonable accommodations can be requested for disabilities or religious needs.", "Submit a request via HR portal."],
    ["Are wellness programs available?", "Employee Engagement: Wellness programs include fitness subsidies and mental health workshops.", "Yes, fitness subsidies and mental health workshops."],
    ["How can I provide feedback anonymously?", "Employee Engagement: Anonymous feedback can be submitted through the HR portal.", "Via the HR portal anonymously."],
    ["What is the deadline for mandatory compliance training?", "Compliance: All employees must complete compliance training annually.", "By December 31 each year."],
    ["How do I update my personal details?", "Employee Records: Personal details can be updated via the HR portal.", "Via the HR portal."],
    ["What is the policy on overtime pay?", "Compensation: Overtime is paid at 1.5x for eligible employees.", "1.5x pay for eligible employees."],
    ["Can I request a flexible work schedule?", "Work Schedule: Flexible schedules are available with manager approval.", "Yes, with manager approval."],
    ["What is the process for promotion requests?", "Career Development: Promotion requests are reviewed during appraisals.", "Submit during annual appraisals."],
    ["Are there employee recognition programs?", "Employee Engagement: TechTrend has quarterly recognition awards.", "Yes, quarterly recognition awards."],
    ["How do I access the Employee Assistance Program?", "Employee Wellness: EAP is accessible via HR portal or direct contact.", "Via HR portal or direct contact."],
    ["What is the policy on workplace safety?", "Workplace Safety: TechTrend adheres to OSHA standards and conducts regular safety training.", "Adheres to OSHA standards with regular training."],
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