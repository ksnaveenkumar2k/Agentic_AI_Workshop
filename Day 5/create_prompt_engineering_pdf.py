
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Define the output PDF file path
pdf_path = "Prompt_Engineering.pdf"

# Define the study material content
study_material = """
<h1>Prompt Engineering for Agents</h1>
<p>Prompt engineering involves designing and refining inputs to language models to achieve desired outputs. In the context of agents, prompt engineering allows for better control over how an agent interacts with the environment and solves specific tasks. This is particularly useful in domains like robotics and conversational AI. By adjusting the structure and content of the prompts, users can enhance an agent's performance on specific tasks. Effective prompt engineering can lead to more accurate responses, improved task efficiency, and better alignment with user intentions.</p>
<p>Key aspects of prompt engineering include crafting clear instructions, providing context, and using examples to guide the model's behavior. For instance, in conversational AI, a well-designed prompt can help an agent maintain a coherent and contextually relevant dialogue. In robotics, prompts can specify task constraints or goals, enabling precise actions.</p>
"""

# Create the PDF
def create_study_material_pdf():
    # Initialize the PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create a list to hold the flowables (PDF elements)
    elements = []
    
    # Add the study material as paragraphs
    for paragraph in study_material.split('\n'):
        if paragraph.strip():
            if paragraph.startswith('<h1>'):
                # Handle heading
                text = paragraph.replace('<h1>', '').replace('</h1>', '')
                elements.append(Paragraph(text, styles['Heading1']))
            elif paragraph.startswith('<p>'):
                # Handle paragraph
                text = paragraph.replace('<p>', '').replace('</p>', '')
                elements.append(Paragraph(text, styles['BodyText']))
            elements.append(Spacer(1, 12))  # Add spacing between elements
    
    # Build the PDF
    doc.build(elements)
    print(f"PDF created successfully at: {pdf_path}")

# Run the function to create the PDF
if __name__ == "__main__":
    try:
        create_study_material_pdf()
    except Exception as e:
        print(f"Error creating PDF: {str(e)}")