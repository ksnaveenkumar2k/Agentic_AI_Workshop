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

# Agentic_AI_Workshop - Day 5 & Module-6


# Task Name - Project: Study Assistant for Quiz Question Generation

**Last Updated:** 13/6/2025

## Overview

The Study Assistant with Gemini AI is a web application designed to help students study more effectively. Users can upload a PDF file (e.g., study material or lecture notes), and the application will:

Extract text from the PDF.

Generate a concise summary of the content in 3-5 bullet points.

Create 3 multiple-choice quiz questions based on the summary, with answers displayed in the format Answer: b) Correct answer.

The application uses the Gemini API (gemini-1.5-flash) for natural language processing tasks like summarization and question generation. It is built using Streamlit for the web interface, PyPDF2 for PDF text extraction, and LangChain for managing prompts and API interactions.

This project is ideal for students who want to quickly review key concepts and test their understanding through quizzes.

**Prerequisites**

Before running the project, ensure you have the following:

Python: Version 3.8 or higher.

Google API Key: You need a Google API key to use the Gemini API. You can obtain one from Google AI Studio.

PDF File: A text-based PDF file to upload (e.g., Prompt_Engineering.pdf).

**Requirements**

## Installation

Install the necessary dependencies:

```bash
pip install --upgrade PyPDF2 langchain langchain-community google-generativeai langchain-google-genai python-dotenv streamlit

Set Up the .env File:

create env [ GOOGLE_API_KEY= Your Api key]

Run Commands

streamlit run study_assistant_gemini.py

```
# Agentic_AI_Workshop - Day 5 & Module-7

# Task Name - Project: Build an Intelligent Travel Assistant AI

**Last Updated:** 13/6/2025

## Overview

A powerful AI-powered travel assistant that provides comprehensive information about any destination worldwide, including weather forecasts, top attractions, best times to visit, and travel tips.

## Features

- **Real-time Weather Data**: Get current weather conditions for any destination using WeatherAPI.com
- **Top Attractions**: Discover the most popular tourist attractions at your destination
- **Best Time to Visit**: Learn about seasonal information and climate details
- **Travel Tips**: Get insights on local customs, safety advice, and transportation options
- **Caching System**: Efficient caching to avoid rate limits and improve performance
- **User-friendly Interface**: Clean, tabbed interface built with Streamlit

## Installation


Install the necessary dependencies:

```bash

1. Navigate to the project directory:
```bash
cd "IHUB_TASK_Agentic_AI_Workshop/Day 5/Module7"

pip install langchain langchain-google-genai langchain-community python-dotenv requests streamlit duckduckgo-search

Set Up the .env File:

create env [ GOOGLE_API_KEY= Your Api key , WEATHER_API_KEY= Your Api key]

weather api key = [ https://www.weatherapi.com/]

Run Commands

streamlit run travel_assistant_streamlit.py

```

# Agentic_AI_Workshop - Day 6 & Module-10

# Task Name - Project: Creating a Conversation AI

**Last Updated:** 16/6/2025

## Overview

This application is a multi-agent AI system built with Streamlit and LangGraph to provide comprehensive market and investment analysis. It analyzes competitor landscapes, market dynamics, and investment potential for specified business categories in given locations. The system uses OpenStreetMap data via Nominatim and Overpass APIs to gather real-world competitor information and generates detailed reports with visualizations.

## Features

**Competitor Analysis:** Identifies nearby competitors, estimates footfall, revenue, and market share

**Market Intelligence:** Calculates market size, growth rate, saturation, and opportunity scores

**Investment Analysis:** Provides ROI projections, payback periods, risk assessments, and recommendations

**Interactive UI:** Streamlit-based interface with downloadable reports and visualizations

**LangGraph Workflow:** Orchestrates analysis through distinct agent nodes for modularity and scalability

## Installation

Install the necessary dependencies:


```bash

pip install streamlit requests pandas plotly numpy langgraph typing-extensions
pip install langgraph==0.2.38

Set Up the .env File:

create env  [ GEMINI_API_KEY = Your Api key]

Run Commands
 cd .\Agentic_AI_Workshop\Day 6>

streamlit run .\langgraph_competitor_analysis.py --server.port 8502


```

# Agentic_AI_Workshop - Hackathon

# Task Name - HR Copilot for Daily Ops

**Last Updated:** 16/6/2025

# HR RAG QA System with AI Agents

This project is an AI-powered HR Query Assistant designed to automate responses to repetitive HR queries, reducing the workload on HR teams. It uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions based on HR policy documents, classifies query intents, triggers mock actions, and escalates sensitive queries (e.g., mental health leave) to human HR staff with conversation context.

## Problem Statement
HR teams spend significant time on repetitive queries and manual administrative tasks, limiting their capacity for strategic initiatives. This system addresses this by:
- Accepting queries via a simulated Slack/WhatsApp interface.
- Classifying intents (e.g., leave, appraisal, payslip, interview, escalate).
- Fetching answers from HR policy PDFs using RAG.
- Triggering mock actions (e.g., appraisal reminders, interview scheduling).
- Escalating sensitive queries with full conversation context.

## Features
- **Query Interface**: Simulated Slack/WhatsApp input box for HR queries.
- **Intent Classification**: Identifies query intent (leave, appraisal, payslip, interview, escalate) using keyword-based rules.
- **RAG Pipeline**: Extracts answers from HR policy PDFs using FAISS and SentenceTransformer embeddings.
- **Mock Actions**: Simulates HR tasks like sending appraisal reminders or scheduling interviews.
- **Escalation**: Forwards sensitive queries (e.g., mental health leave) to HR with conversation history.

```bash

pip install streamlit fitz faiss-cpu numpy langchain-google-genai sentence-transformers python-dotenv

Set Up the .env File:

create env  [ GEMINI_API_KEY = Your Api key]

Run Commands

 cd .\Agentic_AI_Workshop\Hackathon\

 streamlit run "HR Copilot.py"   

```


# Agentic_AI_Workshop - Hackathon

# Task Name - HR Copilot for Daily Ops [Using Multiple Agent]

**Last Updated:** 17/6/2025

## Overview

HR Copilot streamlines HR processes by answering policy queries, automating routine tasks, and escalating sensitive issues. It uses a RAG-based system to retrieve information from a PDF dataset (`TechTrend_Dataset.pdf`), powered by FAISS and Sentence Transformers. The application is built with Streamlit for a user-friendly interface and LangGraph for orchestrating four specialized agents.

## Features

- **Query Handling**: Answers HR questions (e.g., leave policies, compliance deadlines) using RAG.
- **Intent Detection**: Classifies queries as Operational, Sensitive, or Compliance.
- **Task Automation**:
  - Leave approval
  - Onboarding
  - Compliance reminders
  - Personal details updates
  - Employee recognition nominations
- **Human Escalation**: Routes sensitive issues (e.g., harassment) to HR leads.
- **Source Attribution**: Displays retrieved policy chunks with source and chunk ID.
- **Scalable Workflow**: Modular LangGraph architecture for adding new agents or tools.

## Agents

HR Copilot includes **four agents**, each handling a specific function:
1. **Intent Detection Agent**: Classifies queries into Operational, Sensitive, or Compliance using Gemini LLM.
2. **Knowledge Retrieval Agent**: Retrieves relevant policy chunks from `TechTrend_Dataset.pdf` using FAISS and Sentence Transformers.
3. **Task Automation Agent**: Automates workflows for Operational and Compliance queries (e.g., leave approval, compliance nudges).
4. **Human Escalation Agent**: Escalates Sensitive queries to HR leads with summarized context.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- Internet connection for package downloads and Gemini API
- Google Gemini API key

### Dependencies
Install the required Python packages:

```bash

pip install -U streamlit pymupdf faiss-cpu sentence-transformers google-generativeai langchain langgraph langchain-community langchain-google-genai python-dotenv reportlab

Set Up the .env File:

create env  [ GEMINI_API_KEY = Your Api key]

Run Commands

 cd .\Agentic_AI_Workshop\Hackathon\

 streamlit run "HR Copilot.py"   

```


--------------------//----------------//-----------------------------//-------------------------------------//
