import fitz  # PyMuPDF

# Updated PDF path
pdf_path = "D:/ihub_task_agentic_ai/Agentic_AI_Workshop/Hackathon/new_19_hackathon/TechTrend_Dataset.pdf"
output_txt_path = "D:/ihub_task_agentic_ai/Agentic_AI_Workshop/Hackathon/new_19_hackathon/TechTrend_Dataset_Extracted.txt"

# Open the PDF
doc = fitz.open(pdf_path)

# Extract text from each page
all_text = ""
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    text = page.get_text()
    all_text += f"\n--- Page {page_num + 1} ---\n{text}"

# Save to .txt file
with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write(all_text)

doc.close()

print(f"✅ Text saved to: {output_txt_path}")
