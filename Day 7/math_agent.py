# math_agent.py
import re
import os
from typing import TypedDict, Literal, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
import streamlit as st
import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Custom Mathematical Functions
def plus(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b

def multiply(a: float, b: float) -> float:
    return a * b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b

# Define Tools
math_tools = [
    Tool(name="plus", func=plus, description="Add two numbers. Input: two numbers (e.g., '5, 3')."),
    Tool(name="subtract", func=subtract, description="Subtract two numbers. Input: '5, 3'."),
    Tool(name="multiply", func=multiply, description="Multiply two numbers. Input: '5, 3'."),
    Tool(name="divide", func=divide, description="Divide two numbers. Input: '5, 3'."),
]

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ.get("GOOGLE_API_KEY"))
llm_with_tools = llm.bind_tools(math_tools)

# Define State
class AgentState(TypedDict):
    query: str
    is_math: bool
    tool_call: Optional[dict]
    response: str

# Helper to detect operation
def detect_operation(query: str, a: float, b: float) -> str:
    if any(word in query for word in ["subtract", "-"]): return "subtract"
    if any(word in query for word in ["multiply", "times", "*"]): return "multiply"
    if any(word in query for word in ["divide", "divided by", "/"]): return "divide"
    return "plus"

# Chatbot Node
def chatbot_node(state: AgentState) -> AgentState:
    query = state["query"].lower()
    math_patterns = [r"\d+\s*(?:\+|\-|\*|\/|\b(?:plus|subtract|multiply|times|divide|divided by)\b)\s*\d+"]
    state["is_math"] = any(re.search(pattern, query) for pattern in math_patterns)

    if state["is_math"]:
        numbers = re.findall(r"\d+", query)
        if len(numbers) != 2:
            state["response"] = "Please provide exactly two numbers."
            state["is_math"] = False
            state["tool_call"] = None
            return state
        
        a, b = map(float, numbers)
        state["tool_call"] = {"name": detect_operation(query, a, b), "args": {"a": a, "b": b}}
    else:
        state["response"] = llm.invoke(query).content
        state["tool_call"] = None
    return state

# Tools Node
def tools_node(state: AgentState) -> AgentState:
    tool_name = state["tool_call"]["name"]
    args = state["tool_call"]["args"]
    tool = next((t for t in math_tools if t.name == tool_name), None)
    state["response"] = tool.func(**args) if tool else "Unknown operation."
    return state

# Routing
def route_to_tools(state: AgentState) -> Literal["tools", "__end__"]:
    return "tools" if state["is_math"] and state["tool_call"] else "__end__"

# Build and Compile Graph
workflow = StateGraph(AgentState)
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", tools_node)
workflow.add_conditional_edges("chatbot", route_to_tools, {"tools": "tools", "__end__": END})
workflow.add_edge("tools", END)
workflow.set_entry_point("chatbot")
graph = workflow.compile()

# Run Agent
def run_agent(query: str) -> str:
    state = {"query": query, "is_math": False, "tool_call": None, "response": ""}
    return graph.invoke(state)["response"]

# Streamlit Frontend
st.title("Math & General Query Agent")

# Input query
query = st.text_input("Enter your query (e.g., 'What is 10 + 5?' or 'What time is it?')")

# Button to submit query
if st.button("Submit"):
    if query:
        with st.spinner("Processing..."):
            response = run_agent(query)
            st.success("Response:")
            st.write(response)
    else:
        st.error("Please enter a query.")

# Display current date and time
current_time = datetime.datetime.now().strftime("%I:%M %p IST, %B %d, %Y")  # 09:21 AM IST, June 18, 2025
st.caption(f"Current time: {current_time}")

if __name__ == "__main__":
    # Test queries (run via Python if not using Streamlit)
    test_queries = ["What is 10 + 5?", "How much is 20 / 4?", "What time is it?", "6 * 3", "15 - 9"]
    for query in test_queries:
        print(f"Query: {query}")
        print(f"Response: {run_agent(query)}\n")