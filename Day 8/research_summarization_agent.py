
import re
import os
import time
from typing import TypedDict, Literal, Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
import streamlit as st
import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from google.api_core.exceptions import ResourceExhausted
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tavily import TavilyClient

# Load environment variables from .env file
load_dotenv()
print("Environment variables loaded")

# API Keys
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
if not GOOGLE_API_KEY or not TAVILY_API_KEY:
    print("Warning: Missing API key(s) in .env file")
print("API keys checked")

# PDF Path for RAG
PDF_PATH = "D:/ihub_task_agentic_ai/Agentic_AI_Workshop/Day 8/sample_data.pdf"  # Ensure this PDF exists
print(f"PDF path set to: {PDF_PATH}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            print("PDF text extracted successfully")
            return text
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return ""

# Store PDF content as knowledge base and prepare embeddings
PDF_CONTENT = extract_text_from_pdf(PDF_PATH)
if not PDF_CONTENT:
    print("Warning: PDF content is empty, RAG may not work")
sentences = re.split(r'(?<=[.!?])\s+', PDF_CONTENT)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained embedding model
embeddings = model.encode(sentences, convert_to_tensor=False)  # Generate embeddings
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # FAISS index for L2 distance
index.add(np.array(embeddings))  # Add embeddings to index
print("Embeddings and FAISS index prepared")

# Initialize Tavily Client
try:
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    print("Tavily client initialized successfully")
except Exception as e:
    print(f"Error initializing Tavily client: {str(e)}")
    tavily = None  # Fallback to mock if API fails

# Custom Functions
def web_research(query: str) -> str:
    # Real-time web research using Tavily API with fallback
    try:
        current_time = datetime.datetime.now().strftime("%I:%M %p IST, %B %d, %Y")  # 12:50 PM IST, June 18, 2025
        if tavily:
            response = tavily.search(query=query, max_results=3, search_depth="advanced")
            if response and response.get("results"):
                results = [f"{r['title']}: {r['content'][:200]}..." for r in response["results"]]
                return f"Latest information as of {current_time}: {results}"
        return f"Latest information as of {current_time}: No relevant web information available."
    except Exception as e:
        print(f"Web research error: {str(e)}")
        current_time = datetime.datetime.now().strftime("%I:%M %p IST, %B %d, %Y")
        return f"Error fetching web data: {str(e)}. Using mock fallback: Latest information as of {current_time}: No data available."

def rag_retrieve(query: str) -> str:
    # Embedding-based retrieval with FAISS
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), k=1)  # Retrieve top 1 match
    if distances[0][0] < 2.0:  # Threshold for relevance (adjust as needed)
        return sentences[indices[0][0]].strip()
    return "No relevant data found in knowledge base."

def summarize(text: str) -> str:
    # Enhanced summarization with refined prompt and error handling
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
        prompt = f"""Summarize the following text in 1-2 concise sentences, focusing on key facts with a neutral tone. 
        Ensure the summary is clear, accurate, and avoids unnecessary details. If the text is minimal, provide a brief overview.
        Text: {text}
        Summary:"""
        response = llm.invoke(prompt).content
        return response if response else "Summary not available."
    except ResourceExhausted as e:
        return f"Error: API quota exceeded. Please check your plan at https://ai.google.dev/gemini-api/docs/rate-limits. Retry after a few seconds."
    except Exception as e:
        return f"Error during summarization: {str(e)}"

# Define Tools
web_tool = Tool(name="web_research", func=web_research, description="Fetch latest information from the web using Tavily API.")
rag_tool = Tool(name="rag_retrieve", func=rag_retrieve, description="Retrieve information from a knowledge base.")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
llm_with_tools = llm.bind_tools([web_tool, rag_tool])
print("LLM and tools initialized")

# Define State
class AgentState(TypedDict):
    query: str
    route: Literal["llm", "rag", "web_research"]
    intermediate_result: str
    final_summary: str

# Enhanced Router Node
def router_node(state: AgentState) -> AgentState:
    query = state["query"].lower()
    # Check for web research-specific queries
    web_keywords = ["weather", "news", "events", "updates"]
    if ("latest" in query or "current" in query) and any(keyword in query for keyword in web_keywords):
        state["route"] = "web_research"
    # Check for RAG-relevant queries
    elif any(keyword in PDF_CONTENT.lower() for keyword in query.split()):
        state["route"] = "rag"
    # Default to LLM for general or unclear queries
    else:
        state["route"] = "llm"
    state["intermediate_result"] = ""
    state["final_summary"] = ""
    return state

# LLM Node
def llm_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["query"]).content
    state["intermediate_result"] = response
    return state

# RAG Node
def rag_node(state: AgentState) -> AgentState:
    result = rag_retrieve(state["query"])
    state["intermediate_result"] = result
    return state

# Web Research Node
def web_research_node(state: AgentState) -> AgentState:
    result = web_research(state["query"])
    state["intermediate_result"] = result
    return state

# Summarization Node (Updated to ensure summarization)
def summarization_node(state: AgentState) -> AgentState:
    if state["intermediate_result"]:
        summary = summarize(state["intermediate_result"] or "No data provided")
        state["final_summary"] = summary
    else:
        state["final_summary"] = "No data to summarize."
    return state

# Routing Logic
def route_to_agent(state: AgentState) -> Literal["llm", "rag", "web_research", "summarization", "__end__"]:
    if not state.get("route"):
        return "router"
    if state["route"] in ["llm", "rag", "web_research"]:
        return state["route"]
    if state["intermediate_result"]:
        return "summarization"
    return "__end__"

# Build and Compile Graph
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("llm", llm_node)
workflow.add_node("rag", rag_node)
workflow.add_node("web_research", web_research_node)
workflow.add_node("summarization", summarization_node)

# Define edges
workflow.add_conditional_edges("router", route_to_agent)
workflow.add_conditional_edges("llm", lambda state: "summarization")
workflow.add_conditional_edges("rag", lambda state: "summarization")
workflow.add_conditional_edges("web_research", lambda state: "summarization")
workflow.add_edge("summarization", END)
workflow.set_entry_point("router")
graph = workflow.compile()
print("Graph compiled successfully")

# Run Agent with delay to respect rate limits
def run_agent(query: str) -> str:
    state = {"query": query, "route": None, "intermediate_result": "", "final_summary": ""}
    result = graph.invoke(state)
    time.sleep(2)  # Respect retry delay of 2 seconds
    return result["final_summary"] or result["intermediate_result"] or "No response available."

# Streamlit Frontend
st.title("Research and Summarization Agent")

# Input query
query = st.text_input("Enter your query (e.g., 'What is the latest news?' or 'Tell me about history')")

# Button to submit query
if st.button("Submit"):
    if query:
        with st.spinner("Processing..."):
            response = run_agent(query)
            state = {"query": query, "route": None, "intermediate_result": "", "final_summary": ""}
            result = graph.invoke(state)
            agent_used = result["route"] or "LLM"  # Default to LLM if route not set
            st.success("Response:")
            st.write(response)
            st.write(f"Agent: {agent_used}")
    else:
        st.error("Please enter a query.")

# Display current date and time
current_time = datetime.datetime.now().strftime("%I:%M %p IST, %B %d, %Y")  # 12:50 PM IST, June 18, 2025
st.caption(f"Current time: {current_time}")

if __name__ == "__main__":
    # Test queries with delay between calls
    test_queries = [
        "What is the latest news?",
    ]
    for query in test_queries:
        print(f"Query: {query}")
        print(f"Response: {run_agent(query)}\n")