import os
import streamlit as st
import fitz
import faiss
import numpy as np
import google.generativeai as genai
import re
from sentence_transformers import SentenceTransformer
from typing import List, Dict, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
import operator
import json
import uuid

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Streamlit page configuration
st.set_page_config(page_title="HR Copilot with AI Agents", layout="wide")

# Define the path for the PDF
PDF_FILE = "D:/ihub_task_agentic_ai/Agentic_AI_Workshop/Hackathon/new_19_hackathon/TechTrend_Dataset.pdf"

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

model = load_embedding_model()

# Define tools for task automation
@tool
def initiate_leave_approval(employee_id: str, leave_type: str, days: int) -> str:
    """Initiate a leave approval workflow."""
    return f"Leave approval workflow initiated for {employee_id}: {leave_type} for {days} days."

@tool
def initiate_onboarding(employee_id: str, start_date: str) -> str:
    """Initiate an onboarding workflow."""
    return f"Onboarding workflow initiated for {employee_id} starting {start_date}."

@tool
def send_compliance_nudge(employee_id: str, training_type: str) -> str:
    """Send a compliance training reminder to an employee."""
    return f"Compliance nudge sent to {employee_id} for {training_type}."

@tool
def update_personal_details(employee_id: str, field: str, value: str) -> str:
    """Update an employee's personal details in the HR system."""
    return f"Updated {field} to {value} for {employee_id}."

@tool
def initiate_recognition_nomination(employee_id: str, award_type: str) -> str:
    """Nominate an employee for a recognition award."""
    return f"Recognition nomination initiated for {employee_id}: {award_type}."

# Define agent state
class AgentState(TypedDict):
    query: str
    intent: str
    context_chunks: List[Dict]
    actions: List[str]
    escalation_needed: bool
    messages: Annotated[list, operator.add]
    execution_status: Dict[str, str]  # Tracks status of each agent
    final_answer: str  # Stores the final answer for display

# Intent Detection Agent
class IntentDetectionAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY")
        )

    def detect_intent(self, state: AgentState) -> AgentState:
        """Classify query into operational, sensitive, or compliance intent."""
        query = state["query"]
        prompt = f"""
        You are an HR intent detection expert. Classify the following query into one of:
        - Operational (e.g., leave policies, payslips, onboarding, personal details)
        - Sensitive (e.g., harassment, mental health, disputes, accommodations)
        - Compliance (e.g., training deadlines, workplace safety)
        Provide the classification as a single word: 'Operational', 'Sensitive', or 'Compliance'.
        Query: {query}
        """
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            intent = response.content.strip()
            state["intent"] = intent
            state["messages"].append(AIMessage(content=f"Intent detected: {intent}"))
            state["execution_status"]["intent_detection"] = "Success"
        except Exception as e:
            state["intent"] = "Operational"
            state["messages"].append(AIMessage(content=f"Intent detection error: {str(e)}. Defaulting to Operational"))
            state["execution_status"]["intent_detection"] = f"Error: {str(e)}"
        return state

# Knowledge Retrieval Agent
class KnowledgeRetrievalAgent:
    def __init__(self):
        self.index = None
        self.texts = None
        self.chunks = None

    def load_pdf(self, pdf_path: str) -> List[Dict]:
        """Load and extract text from a PDF file."""
        documents = []
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = fitz.open(file)
                text = ""
                for page_num in range(len(pdf_reader)):
                    page = pdf_reader[page_num]
                    text += page.get_text()
                documents.append({"text": text, "source": os.path.basename(pdf_path)})
        except FileNotFoundError:
            st.warning(f"PDF not found: {pdf_path}. Skipping this file.")
        except Exception as e:
            st.warning(f"Error loading {pdf_path}: {str(e)}. Skipping this file.")
        return documents

    def chunk_documents(self, documents: List[Dict], chunk_size=5) -> List[Dict]:
        """Chunk documents by sentences."""
        chunks = []
        for doc in documents:
            text = doc["text"]
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for i in range(0, len(sentences), chunk_size):
                chunk_text = " ".join(sentences[i : i + chunk_size])
                if len(chunk_text.strip()) > 0:
                    chunks.append(
                        {
                            "text": chunk_text,
                            "source": doc["source"],
                            "chunk_id": f"{doc['source']}_chunk_{i // chunk_size}",
                        }
                    )
        return chunks

    def create_faiss_index(self, chunks: List[Dict]):
        """Create a FAISS index from document chunks."""
        self.texts = [chunk["text"] for chunk in chunks]
        embeddings = model.encode(self.texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))
        self.chunks = chunks
        return self.index, self.texts

    def retrieve_context(self, state: AgentState) -> AgentState:
        """Retrieve relevant chunks based on the query."""
        if not self.index or not self.texts or not self.chunks:
            documents = self.load_pdf(PDF_FILE)
            if not documents:
                state["messages"].append(AIMessage(content="Error: No text extracted from PDF."))
                state["execution_status"]["knowledge_retrieval"] = "Error: No text extracted"
                return state
            self.chunks = self.chunk_documents(documents)
            self.index, self.texts = self.create_faiss_index(self.chunks)

        query = state["query"]
        query_vec = model.encode([query])
        distances, indices = self.index.search(np.array(query_vec), k=5)
        context_chunks = [
            {
                "text": self.texts[i],
                "source": self.chunks[i]["source"],
                "chunk_id": self.chunks[i]["chunk_id"],
            }
            for i in indices[0]
        ]
        state["context_chunks"] = context_chunks
        state["messages"].append(AIMessage(content=f"Retrieved {len(context_chunks)} context chunks"))
        state["execution_status"]["knowledge_retrieval"] = "Success"
        return state

# Task Automation Agent
class TaskAutomationAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY")
        )

    def automate_task(self, state: AgentState) -> AgentState:
        """Initiate workflows based on intent and context."""
        if state["intent"] not in ["Operational", "Compliance"]:
            state["messages"].append(AIMessage(content="Non-operational or non-compliance intent detected. Skipping automation."))
            state["execution_status"]["task_automation"] = "Skipped: Invalid intent"
            return state

        query = state["query"]
        context = "\n".join([chunk["text"] for chunk in state["context_chunks"]])
        prompt = f"""
        Based on the query and context, determine if an automated workflow should be initiated.
        Query: {query}
        Context: {context}
        Available workflows: leave_approval, onboarding, compliance_nudge, update_personal_details, recognition_nomination
        If applicable, suggest a workflow and parameters in JSON format.
        Respond with: {{'workflow': 'name', 'parameters': {{'key': 'value'}}}} or {{'workflow': None}}
        Examples:
        - For leave: {{'workflow': 'leave_approval', 'parameters': {{'employee_id': '123', 'leave_type': 'annual', 'days': 5}}}}
        - For compliance: {{'workflow': 'compliance_nudge', 'parameters': {{'employee_id': '123', 'training_type': 'DEI'}}}}
        """
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)
            if result["workflow"]:
                if result["workflow"] == "leave_approval":
                    action = initiate_leave_approval(**result["parameters"])
                elif result["workflow"] == "onboarding":
                    action = initiate_onboarding(**result["parameters"])
                elif result["workflow"] == "compliance_nudge":
                    action = send_compliance_nudge(**result["parameters"])
                elif result["workflow"] == "update_personal_details":
                    action = update_personal_details(**result["parameters"])
                elif result["workflow"] == "recognition_nomination":
                    action = initiate_recognition_nomination(**result["parameters"])
                else:
                    action = "No valid workflow found."
                state["actions"].append(action)
                state["messages"].append(AIMessage(content=action))
                state["execution_status"]["task_automation"] = "Success"
            else:
                state["messages"].append(AIMessage(content="No automation required."))
                state["execution_status"]["task_automation"] = "No action needed"
        except Exception as e:
            state["messages"].append(AIMessage(content=f"Automation error: {str(e)}"))
            state["execution_status"]["task_automation"] = f"Error: {str(e)}"
        return state

# Human Escalation Agent
class HumanEscalationAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY")
        )

    def check_escalation(self, state: AgentState) -> AgentState:
        """Check if query requires human escalation."""
        if state["intent"] == "Sensitive":
            prompt = f"""
            The query '{state["query"]}' was classified as sensitive. Summarize the issue and prepare a message for the HR lead with conversation history.
            History: {state["messages"]}
            """
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                state["escalation_needed"] = True
                state["messages"].append(AIMessage(content=f"Escalation to HR lead: {response.content}"))
                state["execution_status"]["human_escalation"] = "Success"
            except Exception as e:
                state["messages"].append(AIMessage(content=f"Escalation error: {str(e)}"))
                state["execution_status"]["human_escalation"] = f"Error: {str(e)}"
        else:
            state["escalation_needed"] = False
            state["execution_status"]["human_escalation"] = "Not required"
        return state

# Agent Executor
def agent_executor(state: AgentState) -> AgentState:
    """Orchestrates the execution of agents based on state."""
    try:
        if not state["intent"]:
            state["messages"].append(AIMessage(content="No intent detected. Stopping execution."))
            state["execution_status"]["agent_executor"] = "Error: No intent"
            return state

        if state["escalation_needed"]:
            state["messages"].append(AIMessage(content="Query escalated. Stopping further automation."))
            state["execution_status"]["agent_executor"] = "Escalation triggered"
            return state

        # Generate final answer if no escalation
        if state["context_chunks"] and not state["escalation_needed"]:
            context = "\n".join([chunk["text"] for chunk in state["context_chunks"]])
            prompt = f"""
            Based on the context, provide a concise and accurate answer to the query.
            Query: {state["query"]}
            Context: {context}
            """
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
            answer = llm.invoke([HumanMessage(content=prompt)]).content
            state["final_answer"] = answer
            state["messages"].append(AIMessage(content=f"Final answer generated: {answer}"))
            state["execution_status"]["agent_executor"] = "Success"
        else:
            state["final_answer"] = "No answer generated due to insufficient context or escalation."
            state["execution_status"]["agent_executor"] = "No answer generated"
    except Exception as e:
        state["messages"].append(AIMessage(content=f"Executor error: {str(e)}"))
        state["execution_status"]["agent_executor"] = f"Error: {str(e)}"
    return state

# Define LangGraph workflow
def create_workflow():
    workflow = StateGraph(AgentState)
    
    intent_agent = IntentDetectionAgent()
    retrieval_agent = KnowledgeRetrievalAgent()
    automation_agent = TaskAutomationAgent()
    escalation_agent = HumanEscalationAgent()

    # Add nodes
    workflow.add_node("detect_intent", intent_agent.detect_intent)
    workflow.add_node("retrieve_context", retrieval_agent.retrieve_context)
    workflow.add_node("automate_task", automation_agent.automate_task)
    workflow.add_node("check_escalation", escalation_agent.check_escalation)
    workflow.add_node("execute_agents", agent_executor)

    # Define edges
    workflow.set_entry_point("detect_intent")
    workflow.add_edge("detect_intent", "retrieve_context")
    workflow.add_edge("retrieve_context", "automate_task")
    workflow.add_edge("automate_task", "check_escalation")
    workflow.add_edge("check_escalation", "execute_agents")
    workflow.add_edge("execute_agents", END)

    return workflow.compile()

# Streamlit UI
def main():
    st.title("HR Copilot for TechTrend Policies")
    st.markdown("Ask HR-related questions based on TechTrend Innovations policies.")

    # Initialize graph
    try:
        graph = create_workflow()
    except Exception as e:
        st.error(f"Failed to initialize workflow: {str(e)}")
        return

    # Query input
    query = st.text_input(
        "Enter your question:", value="How many days of annual leave are provided?"
    )

    # Process button
    if st.button("Get Answer") and query:
        with st.spinner("Processing query..."):
            try:
                # Initialize state
                initial_state = {
                    "query": query,
                    "intent": "",
                    "context_chunks": [],
                    "actions": [],
                    "escalation_needed": False,
                    "messages": [HumanMessage(content=query)],
                    "execution_status": {},
                    "final_answer": ""
                }
                
                # Run workflow
                result = graph.invoke(initial_state)

                # Display results
                st.subheader("Answer")
                if result["escalation_needed"]:
                    st.write("This query requires human attention. It has been escalated to the HR lead.")
                else:
                    st.write(result["final_answer"])

                st.subheader("Actions Taken")
                if result["actions"]:
                    for action in result["actions"]:
                        st.write(f"- {action}")
                else:
                    st.write("No automated actions taken.")

                st.subheader("Agent Execution Status")
                for agent, status in result["execution_status"].items():
                    st.write(f"{agent}: {status}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()