# from reportlab.lib import colors
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
# from reportlab.lib.styles import getSampleStyleSheet

# # Expanded Dataset
# data = [
#     ["Question", "Context", "Answer"],
#     ["How many days of annual leave are provided?", "Leave Policy: Employees at TechTrend Innovations are entitled to 20 days of annual leave per calendar year, accruing monthly.", "20 days of annual leave per year."],
#     ["How long is parental leave?", "Leave Policy: Parental leave at TechTrend Innovations includes maternity and paternity leave, offering up to 12 weeks for primary caregivers.", "Up to 12 weeks."],
#     ["When must leave requests be submitted?", "Leave Policy: All leave requests must be submitted through the HR portal with manager approval.", "At least 7 days in advance via the HR portal."],
#     ["What is the process for emergency leave?", "Leave Policy: Emergency leave can be requested with less than 7 days’ notice in exceptional circumstances, subject to HR approval.", "Submit via HR portal; approval within 24 hours."],
#     ["Can I carry over unused leave?", "Leave Policy: Up to 5 days of unused annual leave can be carried over to the next year with manager approval.", "Up to 5 days with manager approval."],
#     ["When do annual appraisals take place?", "Appraisal Process: Annual appraisals occur in December to review performance and set goals.", "In December."],
#     ["By when must self-assessment forms be submitted?", "Appraisal Process: Employees must submit self-assessment forms to prepare for appraisals.", "By November 15."],
#     ["Are performance bonuses available?", "Appraisal Process: Performance bonuses are awarded based on appraisal outcomes.", "Yes, based on appraisal scores."],
#     ["How is the appraisal score calculated?", "Appraisal Process: Scores are based on KPIs, peer feedback, and manager evaluation.", "Based on KPIs, peer feedback, and manager evaluation."],
#     ["Where can I find my payslip?", "Payslip Access: Payslips are available on the HR portal under the Payroll section.", "On the HR portal under 'Payroll'."],
#     ["Can I get paper payslips?", "Payslip Access: Employees can opt for paper payslips via HR request.", "Yes, employees can opt for paper payslips."],
#     ["How long to resolve payslip discrepancies?", "Payslip Access: Discrepancies must be reported to HR for resolution.", "Contact HR within 30 days."],
#     ["How do I claim travel expenses?", "Expense Claims: Work-related travel expenses are reimbursable with valid receipts.", "Submit claims via the HR portal with receipts."],
#     ["What is the deadline for expense claims?", "Expense Claims: Claims must be submitted within 60 days of incurring expenses.", "Within 60 days."],
#     ["How long for expense approval?", "Expense Claims: Approval process takes 5-10 business days after submission.", "5-10 business days."],
#     ["What expenses are eligible for reimbursement?", "Expense Claims: Eligible expenses include travel, accommodation, and work-related supplies.", "Travel, accommodation, and work-related supplies."],
#     ["How many training hours are offered?", "Training and Development: TechTrend offers 40 hours of professional development annually.", "40 hours annually."],
#     ["How do I enroll in training courses?", "Training and Development: Courses are available via the Learning Portal.", "Via the Learning Portal."],
#     ["Can I attend workshops?", "Training and Development: Workshops count toward annual training hours.", "Yes, workshops are available."],
#     ["Are external certifications reimbursable?", "Training and Development: Approved external certifications are fully reimbursable.", "Yes, with prior approval."],
#     ["How many days can I work remotely?", "Remote Work Policy: Employees may work remotely up to 3 days per week, subject to role requirements.", "Up to 3 days per week."],
#     ["Do remote workers need manager approval?", "Remote Work Policy: Remote work arrangements require manager approval.", "Yes, subject to manager approval."],
#     ["What is required for a remote workspace?", "Remote Work Policy: Remote employees need a dedicated workspace and stable internet.", "A dedicated workspace and virtual check-ins."],
#     ["How long is the onboarding program?", "Onboarding Process: New hires complete a structured onboarding program.", "2 weeks."],
#     ["Do new hires get a buddy?", "Onboarding Process: A buddy is assigned to support new hires.", "Yes, for the first 30 days."],
#     ["What training is included in onboarding?", "Onboarding Process: Onboarding includes system training and team introductions.", "System training and team introductions."],
#     ["Is DEI training mandatory?", "Diversity and Inclusion: TechTrend requires annual DEI training for all employees.", "Yes, annually."],
#     ["Can I join affinity groups?", "Diversity and Inclusion: Affinity groups are open to employees from underrepresented communities.", "Yes, for underrepresented communities."],
#     ["What is TechTrend’s diversity commitment?", "Diversity and Inclusion: TechTrend is committed to fostering a diverse and inclusive workplace.", "Committed to a diverse workplace with DEI initiatives."],
#     ["How do I report a harassment issue?", "Workplace Conduct: Harassment issues must be reported to HR immediately for investigation.", "Report to HR immediately."],
#     ["What support is available for mental health?", "Employee Wellness: TechTrend offers an Employee Assistance Program (EAP) for mental health support.", "Employee Assistance Program (EAP)."],
#     ["How are workplace disputes resolved?", "Workplace Conduct: Disputes are handled through mediation and HR review.", "Through mediation and HR review."],
#     ["What is the process for requesting accommodations?", "Employee Support: Reasonable accommodations can be requested for disabilities or religious needs.", "Submit a request via HR portal."],
#     ["Are wellness programs available?", "Employee Engagement: Wellness programs include fitness subsidies and mental health workshops.", "Yes, fitness subsidies and mental health workshops."],
#     ["How can I provide feedback anonymously?", "Employee Engagement: Anonymous feedback can be submitted through the HR portal.", "Via the HR portal anonymously."],
#     ["What is the deadline for mandatory compliance training?", "Compliance: All employees must complete compliance training annually.", "By December 31 each year."],
#     ["How do I update my personal details?", "Employee Records: Personal details can be updated via the HR portal.", "Via the HR portal."],
#     ["What is the policy on overtime pay?", "Compensation: Overtime is paid at 1.5x for eligible employees.", "1.5x pay for eligible employees."],
#     ["Can I request a flexible work schedule?", "Work Schedule: Flexible schedules are available with manager approval.", "Yes, with manager approval."],
#     ["What is the process for promotion requests?", "Career Development: Promotion requests are reviewed during appraisals.", "Submit during annual appraisals."],
#     ["Are there employee recognition programs?", "Employee Engagement: TechTrend has quarterly recognition awards.", "Yes, quarterly recognition awards."],
#     ["How do I access the Employee Assistance Program?", "Employee Wellness: EAP is accessible via HR portal or direct contact.", "Via HR portal or direct contact."],
#     ["What is the policy on workplace safety?", "Workplace Safety: TechTrend adheres to OSHA standards and conducts regular safety training.", "Adheres to OSHA standards with regular training."],
# ]

# # Create PDF
# pdf_file = "D:/ihub_task_agentic_ai/Agentic_AI_Workshop/Hackathon/TechTrend_Dataset.pdf"
# doc = SimpleDocTemplate(pdf_file, pagesize=letter)
# styles = getSampleStyleSheet()
# elements = []

# # Add title
# title = Paragraph("TechTrend Innovations Policy Dataset", styles['Title'])
# elements.append(title)

# # Create table
# table = Table(data)

# # Simplified table style for better text extraction
# table.setStyle(TableStyle([
#     ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#     ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#     ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
#     ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),  # Use a standard font
#     ('FONTSIZE', (0, 0), (-1, -1), 10),
#     ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
#     ('BACKGROUND', (0, 1), (-1, -1), colors.white),  # Simpler background
#     ('GRID', (0, 0), (-1, -1), 1, colors.black),
#     ('VALIGN', (0, 0), (-1, -1), 'TOP'),
#     ('LEFTPADDING', (0, 0), (-1, -1), 5),
#     ('RIGHTPADDING', (0, 0), (-1, -1), 5),
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
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Expanded Policy Data
policy_data = {
    "Leave Policy": [
        ["Annual Leave", "Employees are entitled to 20 days of annual leave per year, accruing at 1.67 days per month.", "20 days/year; submit via HR portal 7 days in advance; max 5 days carryover with approval."],
        ["Parental Leave", "Primary caregivers receive 12 weeks paid leave; secondary caregivers get 4 weeks.", "Submit via HR portal; 30 days’ notice preferred."],
        ["Emergency Leave", "Available for unforeseen circumstances with no minimum notice.", "Submit via HR portal; approval within 24 hours."],
        ["Sick Leave", "10 days paid sick leave annually, non-accruable.", "Notify manager within 24 hours; medical certificate for 3+ days."],
        ["Bereavement Leave", "Up to 5 days paid leave for immediate family loss.", "Notify HR within 48 hours; documentation may be required."],
        ["Personal Leave", "Up to 3 days unpaid leave for personal matters.", "Submit via HR portal; manager approval required."],
        ["Jury Duty Leave", "Paid leave for jury duty as required by law.", "Provide summons to HR; max 10 days/year."],
    ],
    "Appraisal Process": [
        ["Annual Appraisal", "Held in December to review performance and set goals.", "Submit self-assessment by Nov 15; includes KPIs, peer feedback."],
        ["Performance Bonuses", "Bonuses of 5-15% of salary based on appraisal scores.", "Disbursed in January; tied to KPI achievement."],
        ["Promotion Review", "Evaluated during appraisals based on performance and skills.", "Apply via HR portal in December."],
        ["Mid-Year Review", "Optional mid-year check-in to assess progress.", "Scheduled in June; submit updates via HR portal."],
        ["360-Degree Feedback", "Incorporates feedback from peers, subordinates, and managers.", "Collected via HR portal in November."],
    ],
    "Payroll and Payslips": [
        ["Payslip Access", "Digital payslips available on HR portal under Payroll.", "Access anytime; paper option upon request."],
        ["Payslip Discrepancies", "Report discrepancies to HR for resolution.", "Contact HR within 30 days of issuance."],
        ["Overtime Pay", "1.5x pay for hours beyond 40/week for eligible employees.", "Submit hours via HR portal; non-exempt roles only."],
        ["Payroll Schedule", "Bi-weekly payroll processed on 15th and last day of month.", "Direct deposit; contact HR for issues."],
        ["Tax Withholding", "Employees can adjust tax withholding via HR portal.", "Submit W-4 form updates by Dec 1."],
    ],
    "Expense Claims": [
        ["Travel Expenses", "Reimbursable for work-related travel with valid receipts.", "Submit via HR portal within 60 days."],
        ["Approval Timeline", "Claims processed within 5-10 business days.", "Approval notification via email."],
        ["Eligible Expenses", "Includes travel, accommodation, meals, and supplies.", "Receipts required; pre-approval for non-standard items."],
        ["Mileage Reimbursement", "Reimbursed at $0.58/mile for business travel.", "Submit mileage log via HR portal."],
        ["Expense Limits", "Per diem rates: $50/day for meals, $150/night for lodging.", "Exceeding limits requires VP approval."],
    ],
    "Training and Development": [
        ["Annual Training Hours", "40 hours of professional development annually.", "Access via Learning Portal."],
        ["External Certifications", "Approved certifications fully reimbursable.", "Submit approval request before enrollment."],
        ["Workshops", "Internal/external workshops count toward training hours.", "Register via Learning Portal; manager approval."],
        ["Leadership Training", "Available for employees identified as high-potential.", "Nomination by manager; held quarterly."],
        ["Online Courses", "Access to LinkedIn Learning and Coursera via HR portal.", "Unlimited access within training hours."],
        ["Mentorship Training", "Training for mentors to support new hires.", "Offered biannually; apply via HR portal."],
    ],
    "Remote Work Policy": [
        ["Remote Work Allowance", "Up to 3 days/week remote, role-dependent.", "Requires manager approval; stable internet mandatory."],
        ["Workspace Requirements", "Dedicated workspace and virtual check-ins required.", "Equipment subsidy up to $500."],
        ["Approval Process", "Formalize remote work via HR portal.", "Submit request 14 days in advance."],
        ["Remote Work Tools", "Company provides Zoom, Slack, and VPN access.", "IT setup during onboarding."],
        ["Home Office Stipend", "One-time $200 stipend for home office setup.", "Claim via HR portal within 90 days."],
    ],
    "Onboarding Process": [
        ["Onboarding Program", "2-week program with system training and introductions.", "Begins on start date; buddy assigned."],
        ["Buddy Program", "Buddy supports new hires for 30 days.", "Assigned by HR."],
        ["Onboarding Checklist", "Includes IT setup, policy training, and role onboarding.", "Completed via HR portal."],
        ["New Hire Orientation", "One-day session on company culture and policies.", "Held first Monday of month."],
        ["Role-Specific Training", "Tailored training based on department needs.", "Scheduled within first week."],
    ],
    "Diversity, Equity, and Inclusion (DEI)": [
        ["DEI Training", "Mandatory annual training to foster inclusivity.", "Complete by Dec 31 via Learning Portal."],
        ["Affinity Groups", "Open to underrepresented communities for networking.", "Join via HR portal; quarterly events."],
        ["Diversity Commitment", "Equitable hiring and promotion practices.", "Annual DEI report published."],
        ["Inclusive Leadership", "Training for managers on inclusive practices.", "Offered biannually; mandatory for new managers."],
        ["DEI Metrics", "Tracked to ensure diverse representation.", "Reported in annual DEI report."],
    ],
    "Workplace Conduct and Support": [
        ["Harassment Reporting", "Report harassment to HR for confidential investigation.", "Submit via HR portal or to HR lead."],
        ["Mental Health Support", "Employee Assistance Program (EAP) offers free counseling.", "Access via HR portal or 24/7 hotline."],
        ["Dispute Resolution", "Resolved through mediation and HR review.", "Contact HR within 7 days."],
        ["Accommodations", "Reasonable accommodations for disabilities/religious needs.", "Submit request via HR portal; reviewed in 5 days."],
        ["Whistleblower Policy", "Protects employees reporting misconduct.", "Submit anonymously via HR portal."],
        ["Conflict of Interest", "Employees must disclose potential conflicts.", "Submit disclosure form via HR portal."],
    ],
    "Employee Engagement": [
        ["Wellness Programs", "Fitness subsidies and mental health workshops.", "Subsidies up to $200/year; register via HR portal."],
        ["Recognition Awards", "Quarterly awards with monetary bonuses.", "Nominations via HR portal; announced at town halls."],
        ["Anonymous Feedback", "Submit feedback anonymously to improve policies.", "Via HR portal; reviewed monthly."],
        ["Team Building Events", "Quarterly events to foster collaboration.", "Details announced via internal newsletter."],
        ["Employee Surveys", "Annual surveys to gauge satisfaction.", "Conducted in March; results shared in April."],
    ],
    "Compliance and Safety": [
        ["Compliance Training", "Mandatory annual training on ethics and policies.", "Complete by Dec 31 via Learning Portal."],
        ["Workplace Safety", "Adheres to OSHA standards with regular drills.", "Quarterly safety training; report hazards to HR."],
        ["Data Protection", "Compliance with GDPR and company data policies.", "Training during onboarding."],
        ["Anti-Bribery Policy", "Prohibits bribery and unethical payments.", "Annual training; report violations to HR."],
        ["Health and Safety Audits", "Conducted biannually to ensure compliance.", "Results shared with employees."],
    ],
    "Benefits and Perks": [
        ["Health Insurance", "Comprehensive medical, dental, and vision plans.", "Enroll via HR portal within 30 days of hire."],
        ["Retirement Plan", "401(k) with 4% company match.", "Enroll via HR portal; vesting after 2 years."],
        ["Life Insurance", "Basic life insurance provided at no cost.", "Optional supplemental plans available."],
        ["Commuter Benefits", "Pre-tax commuter benefits for public transit.", "Enroll via HR portal."],
        ["Employee Discounts", "Discounts on company products and partner services.", "Access via HR portal."],
    ],
    "Employee Termination": [
        ["Voluntary Termination", "Employees must provide 2 weeks’ notice.", "Submit resignation via HR portal."],
        ["Involuntary Termination", "Handled by HR with documented cause.", "Exit interview mandatory."],
        ["Severance Policy", "Eligible employees receive severance based on tenure.", "1 week’s pay per year of service."],
        ["Exit Process", "Includes return of company property and final paycheck.", "Completed within 5 days of termination."],
    ],
    "Travel Policy": [
        ["Business Travel", "All travel must be pre-approved by manager.", "Submit travel request via HR portal."],
        ["Travel Booking", "Book through company travel portal for reimbursement.", "Non-portal bookings require VP approval."],
        ["International Travel", "Requires additional insurance and compliance checks.", "Submit 30 days in advance."],
        ["Travel Safety", "Employees must follow company travel safety guidelines.", "Briefing provided before travel."],
    ],
    "Code of Conduct": [
        ["Professional Behavior", "Employees must maintain professionalism in all interactions.", "Violations reported to HR."],
        ["Dress Code", "Business casual attire required in office.", "Casual Fridays allowed."],
        ["Confidentiality", "Employees must protect company and client data.", "Sign NDA during onboarding."],
        ["Social Media Policy", "Personal social media must not disclose company info.", "Violations may lead to disciplinary action."],
    ],
}

# Create PDF
pdf_file = "D:/ihub_task_agentic_ai/Agentic_AI_Workshop/Hackathon/new_19_hackathon/TechTrend_Dataset.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch, leftMargin=0.5*inch, rightMargin=0.5*inch)
styles = getSampleStyleSheet()

# Define custom styles
styles.add(ParagraphStyle(name='SectionHeader', fontName='Helvetica-Bold', fontSize=14, spaceAfter=12, leading=16))
styles.add(ParagraphStyle(name='CoverTitle', fontName='Helvetica-Bold', fontSize=18, spaceAfter=20, alignment=1))
styles.add(ParagraphStyle(name='CoverSubtitle', fontName='Helvetica', fontSize=12, spaceAfter=10, alignment=1))

elements = []

# Cover page
elements.append(Paragraph("TechTrend Innovations", styles['CoverTitle']))
elements.append(Paragraph("HR Policy Report", styles['CoverTitle']))
elements.append(Paragraph("Comprehensive Employee Policy Guide", styles['CoverSubtitle']))
elements.append(Paragraph("Generated on June 19, 2025", styles['CoverSubtitle']))
elements.append(Spacer(1, 0.5*inch))

# Table of contents
elements.append(Paragraph("Table of Contents", styles['SectionHeader']))
toc_data = [["Section", "Page"]]
for i, category in enumerate(policy_data.keys(), start=1):
    toc_data.append([category, str(i+1)])
toc_table = Table(toc_data, colWidths=[400, 100])
toc_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('LEFTPADDING', (0, 0), (-1, -1), 5),
    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
]))
elements.append(toc_table)
elements.append(Spacer(1, 0.3*inch))

# Add policy sections
for category, policies in policy_data.items():
    # Section header
    elements.append(Paragraph(category, styles['SectionHeader']))
    
    # Policy table
    table_data = [["Policy Name", "Description", "Key Details"]] + policies
    table = Table(table_data, colWidths=[150, 250, 150])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.2*inch))

# Build PDF
doc.build(elements)
print(f"PDF file '{pdf_file}' has been created with the expanded policy dataset.")