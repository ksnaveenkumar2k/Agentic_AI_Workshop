# # Agentic_AI_Workshop  Day 2 - Module1 & Module2

# Task Name - stand-up-comedy

**Last Updated:**  - 10/6/2025

# Task Name - Project: Create a Business Report on the Scope of Generative AI

**Last Updated:** - 10/6/2025

# Task Name - Final Report Analysis

**Last Updated:**  - 11/6/2025


--------------------//----------------//-----------------------------//-------------------------------------//

# Agentic_AI_Workshop - Day 3 & Module-3

**Last Updated:** 11/6/2025

---

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for Question Answering (QA) using Streamlit, FAISS, SentenceTransformer, and Google Gemini API.

The system allows users to ask questions about AI research papers (focused on transformer models), retrieves relevant chunks from preloaded PDFs, and generates detailed answers using Gemini.

---

## 📁 Project Structure

IHUB_TASK_AGENTIC_AI/
└── Agentic_AI_Workshop/
└── Day 3/
├── RAG Project Dataset/
│ ├── 1706.03762v7.pdf
│ ├── 2005.11401v4.pdf
│ ├── 2005.14165v4.pdf
│ └── SNS.pdf
├── rag_cache/
├── chat_history.json
├── rag_qa_system.py
└── README.md

## Installation

Install the necessary dependencies:

```bash
pip install streamlit
pip install PyMuPDF
pip install faiss-cpu
pip install numpy
pip install google-generativeai
pip install sentence-transformers

**🚀 Running the App**:

streamlit run rag_qa_system.py

```
--------------------//----------------//-----------------------------//-------------------------------------//

# Agentic_AI_Workshop - Day 4 & Module-4


# Task Name - Project: Healthcare Policy Sales Agent

**Last Updated:** 12/6/2025

## Overview

This assistant is designed to help users find the most suitable health insurance policies based on their personal and family healthcare needs. By collecting key details such as age, family type, number of dependents, and special requirements, the system intelligently matches user profiles with the best-fit insurance plans available in the dataset.

The assistant ensures personalized policy suggestions and clear explanations for each recommendation. It focuses solely on healthcare insurance coverage and is trained to respond only using the provided data.

Link also attached in Healthcare Policy Agent PDF 


# Agentic_AI_Workshop - Day 4 & Module-5

# Task Name - Project: Building a Web Research Agent using the ReAct Pattern

**Last Updated:** 12/6/2025

# ReAct Web Research Agent

A Python-based ReAct (Reasoning + Acting) agent that researches a user-defined topic using the Gemini API for question generation and the Tavily API for web searches, compiling findings into a structured markdown report. This project fulfills the requirements of the "Building AI Agents from Scratch" assignment.

**Project Description:**

This project implements a web research agent following the ReAct pattern. The agent:

Generates 5-6 research questions using Google's Gemini API (Planning Phase).
Searches the web for answers using the Tavily API (Acting Phase).
Compiles a structured markdown report with a title, introduction, question sections, and conclusion.
The assignment requires integrating an LLM (Gemini) and a web search tool (Tavily) to automate research on topics like "Climate Change."

**Features:**

**Question Generation:** Uses Gemini API to create 5-6 diverse, well-structured research questions.
**Web Search:** Retrieves relevant web results (title and content) using Tavily API.
**Report Compilation:** Generates a markdown report (research_report.md) with organized findings.
**Modular Design:** Implements the ReAct pattern via a ReActAgent class.
**Error Handling:** Handles API failures and malformed responses gracefully.

**Requirements**

## Installation

Install the necessary dependencies:

```bash
pip install google-generativeai tavily-python python-dotenv

Set Up Environment Variables:

Create a .env file in the project root:

GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key

Obtain keys from Google AI Studio (Gemini) and Tavily.

Run Commands

To run the script with the default topic ("Climate Change"):

python react_agent_gemini.py

To use a custom topic, modify the topic variable in react_agent_gemini.py:

topic = "Your Topic Here"  # e.g., "Artificial Intelligence"

The script generates a research_report.md file with the research findings.


```

**File Structure**

react_agent_gemini.py: Main script implementing the ReAct agent.

research_report.md: Generated markdown report (output).

.env: Environment file for API keys (not tracked in Git).

--------------------//----------------//-----------------------------//-------------------------------------//
