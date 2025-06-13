import PyPDF2
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
import os
import re
from google.api_core import exceptions
from dotenv import load_dotenv
import streamlit as st
import io

# Load environment variables from .env file
load_dotenv()

# Initialize the Gemini API (assumes GOOGLE_API_KEY is set in .env)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error(
        "GOOGLE_API_KEY is not set in the .env file. Please create a .env file with 'GOOGLE_API_KEY=your-api-key' or obtain a key from https://aistudio.google.com/app/apikey."
    )
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.3,  # Reduced to reduce variability
    max_tokens=1500,  # Increased to avoid truncation
)


# Function to extract text from a PDF file (accepts file bytes for Streamlit upload)
def extract_text_from_pdf(pdf_file):
    try:
        # Read the uploaded file as bytes
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        study_material = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                study_material += text + "\n"
        if not study_material.strip():
            return "Error: No text extracted from PDF."
        return study_material
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"


# Function to clean extracted text
def clean_text(text):
    # Remove extra whitespace, newlines, and special characters
    text = re.sub(r"\s+", " ", text.strip())
    # Remove non-printable characters
    text = "".join(char for char in text if char.isprintable())
    return text


# Prompt template for summarizing content
summary_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
    You are an expert study assistant. Summarize the following study material into 3-5 concise bullet points, focusing on the most important concepts. Ensure clarity and brevity, avoiding unnecessary details.

    Study Material:
    {text}

    Output format:
    - Key concept 1
    - Key concept 2
    - ...
    """,
)

# Prompt template for generating multiple-choice questions (stricter formatting with examples)
quiz_prompt_template = PromptTemplate(
    input_variables=["summary"],
    template="""
    You are an expert study assistant. Based on the following summarized study material, generate exactly 3 multiple-choice questions. Each question MUST:
    - Have a clear, non-empty question text immediately following the question header on the next line.
    - The question text must be a complete, relevant sentence directly related to the summary points provided.
    - Have exactly 4 distinct, plausible options, each on a separate line.
    - Have exactly one correct answer that matches one of the options exactly.
    - Test understanding of the key concepts in the summary.
    - Be clear and relevant to the material.
    - Follow the exact output format specified below with no deviations, including no extra newlines, missing lines, or empty lines.

    Summary:
    {summary}

    Output format MUST be:
    **Question 1:**
    What is the primary goal of prompt engineering in agent-based systems?
    1.a) To optimize agent memory
    2.b) To refine inputs for better output control
    3.c) To improve agent hardware
    4.d) To increase computational power
    **Answer:** To refine inputs for better output control

    **Question 2:**
    In which domain is prompt engineering commonly used?
    1.a) Image recognition
    2.b) Conversational AI
    3.c) Data processing
    4.d) Video editing
    **Answer:** Conversational AI

    **Question 3:**
    Which technique is key to prompt engineering?
    1.a) Upgrading hardware components
    2.b) Providing context in prompts
    3.c) Increasing data storage
    4.d) Automating data collection
    **Answer:** Providing context in prompts

    Ensure there are no empty lines between the question header, question text, options, and answer. Each question block must have exactly 6 lines. The question text must be a complete, relevant sentence based on the summary. If you cannot generate a relevant question, use a fallback question based on the summary, but the question text must never be empty.
    """,
)

# Create RunnableSequences instead of LLMChain
summary_chain = summary_prompt_template | llm
quiz_chain = quiz_prompt_template | llm


# Function to parse quiz output into a structured format (with silent error handling)
def parse_quiz_output(quiz_text):
    questions = []
    quiz_lines = quiz_text.split("\n")
    i = 0
    while i < len(quiz_lines):
        line = quiz_lines[i].strip()
        if line.startswith("**Question"):
            question_data = {}
            # Check if enough lines remain for question, 4 options, and answer (6 lines total)
            if i + 5 >= len(quiz_lines):
                break

            # Extract question text
            i += 1
            question_text = quiz_lines[i].strip()
            if not question_text:
                i += 5  # Skip to the next question
                continue
            question_data["question"] = question_text

            # Extract options and their letters
            options = []
            option_letters = []
            for j in range(4):
                i += 1
                # Ensure the line exists and matches the expected format (e.g., "1.a) Option")
                if i >= len(quiz_lines) or not re.match(
                    r"^\d+\.\w+\)\s", quiz_lines[i]
                ):
                    i += (4 - j) + 1  # Skip remaining options and answer
                    break
                try:
                    # Split "1.a) Option" into letter and text
                    match = re.match(r"^\d+\.(\w+)\)\s(.+)", quiz_lines[i].strip())
                    if not match:
                        i += (4 - j) + 1
                        break
                    letter, option_text = match.groups()
                    options.append(option_text)
                    option_letters.append(letter)
                except IndexError:
                    i += (4 - j) + 1  # Skip remaining options and answer
                    break
            else:
                # Only proceed if all 4 options were parsed successfully
                question_data["options"] = options
                question_data["option_letters"] = option_letters
                # Extract answer
                i += 1
                if i >= len(quiz_lines) or not quiz_lines[i].strip().startswith(
                    "**Answer:**"
                ):
                    continue
                try:
                    answer = quiz_lines[i].strip().split("**Answer:** ")[1]
                    # Find the corresponding letter for the answer
                    answer_letter = None
                    for opt, letter in zip(options, option_letters):
                        if opt == answer:
                            answer_letter = letter
                            break
                    if answer_letter:
                        question_data["answer"] = answer
                        question_data["answer_letter"] = answer_letter
                        questions.append(question_data)
                except IndexError:
                    continue
        i += 1
    return questions


# Main function to process study material and generate quiz
def generate_study_quiz(pdf_file):
    try:
        # Step 1: Extract and clean text from PDF
        study_material = extract_text_from_pdf(pdf_file)
        if "Error" in study_material:
            return {"error": study_material}

        cleaned_material = clean_text(study_material)

        # Step 2: Summarize the content using invoke
        summary_result = summary_chain.invoke(
            {"text": cleaned_material[:10000]}
        )  # Limit input to avoid token limits
        summary = (
            summary_result.content
            if hasattr(summary_result, "content")
            else summary_result
        )
        summary_points = [
            point.strip()
            for point in summary.split("\n")
            if point.strip().startswith("-")
        ]

        # Step 3: Generate quiz questions based on the summary using invoke
        quiz_result = quiz_chain.invoke({"summary": summary})
        quiz = quiz_result.content if hasattr(quiz_result, "content") else quiz_result

        # Debug: Display raw quiz output (temporary)
        st.markdown("### Debug: Raw Quiz Output")
        st.text(quiz)

        quiz_questions = parse_quiz_output(quiz)

        # Step 4: Return structured results
        return {
            "summary": summary_points,
            "quiz_questions": quiz_questions,
            "error": None,
        }
    except exceptions.GoogleAPIError as e:
        return {
            "error": f"Error with Gemini API: {str(e)}. Please verify your API key in the .env file and ensure you're within rate limits at https://aistudio.google.com/app/apikey."
        }
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# Streamlit UI
def main():
    st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="wide")
    st.title("📚 Study Assistant with Gemini AI")
    st.markdown(
        "Upload a PDF file to generate a summary and quiz questions based on its content."
    )

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            result = generate_study_quiz(uploaded_file)

        if "error" in result and result["error"]:
            st.error(result["error"])
        else:
            # Display summary
            st.header("Summary of Study Material")
            for point in result["summary"]:
                st.markdown(point)

            # Display quiz questions (static text format with answers)
            st.header("Quiz Questions")
            if result["quiz_questions"]:
                for idx, question in enumerate(result["quiz_questions"], 1):
                    st.markdown(f"Question {idx}:")
                    st.markdown(f"{question['question']}")
                    for opt_idx, (option, letter) in enumerate(
                        zip(question["options"], question["option_letters"]), 1
                    ):
                        st.markdown(f"{opt_idx}.{letter}) {option}")
                    st.markdown(
                        f"Answer: {question['answer_letter']}) {question['answer']}"
                    )
                    st.markdown("")  # Add a blank line for spacing


if __name__ == "__main__":
    main()
