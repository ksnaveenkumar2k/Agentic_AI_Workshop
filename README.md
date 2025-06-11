# # Agentic_AI_Workshop  Day 2 - Module1 & Module2

# Task Name - stand-up-comedy

**Last Updated:**  - 10/6/2025

# Task Name - Project: Create a Business Report on the Scope of Generative AI

**Last Updated:** - 10/6/2025

# Task Name - Final Report Analysis

**Last Updated:**  - 11/6/2025


--------------------//----------------//-----------------------------//-------------------------------------//

# Agentic_AI_Workshop - Day 3

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

--------------------//----------------//-----------------------------//-------------------------------------//

