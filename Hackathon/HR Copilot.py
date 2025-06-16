import os
import streamlit as st
import fitz
import faiss
import numpy as np
import google.generativeai as genai
import re
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="RAG QA System with AI Agents", layout="wide")

# Define the path for the PDF
PDF_FILE = "D:/ihub_task_agentic_ai/Agentic_AI_Workshop/Hackathon/TechTrend_Dataset.pdf"

# Configure Gemini API using the API key from the environment variables
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")


model = load_embedding_model()


# AI Agent Classes
class DocumentProcessorAgent:
    """Agent responsible for loading and chunking documents."""

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


class IndexerAgent:
    """Agent responsible for creating and managing FAISS index."""

    def __init__(self):
        self.index = None
        self.texts = None

    def create_faiss_index(self, chunks: List[Dict]):
        """Create a FAISS index from document chunks."""
        self.texts = [chunk["text"] for chunk in chunks]
        embeddings = model.encode(self.texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))
        return self.index, self.texts


class RetrieverAgent:
    """Agent responsible for retrieving relevant document chunks."""

    def retrieve_context(
        self, index, texts: List[str], chunks: List[Dict], query: str, k: int = 5
    ) -> List[Dict]:
        """Retrieve relevant chunks based on the query."""
        query_vec = model.encode([query])
        distances, indices = index.search(np.array(query_vec), k)
        return [
            {
                "text": texts[i],
                "source": chunks[i]["source"],
                "chunk_id": chunks[i]["chunk_id"],
            }
            for i in indices[0]
        ]


class AnswerGeneratorAgent:
    """Agent responsible for generating answers using LangChain's ChatGoogleGenerativeAI."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY")
        )

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Dict:
        """Generate an answer using LangChain's ChatGoogleGenerativeAI."""
        context = "\n\n".join(
            [f"Chunk {i+1}:\n{chunk['text']}" for i, chunk in enumerate(context_chunks)]
        )
        prompt = (
            "You are an expert AI assistant. Based only on the following context, "
            "provide a detailed, accurate, and well-cited answer to the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        try:
            # Use LangChain's chat model to generate the response
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"

        sources = [
            {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"][:300] + "...",
            }
            for chunk in context_chunks
        ]
        return {"answer": answer, "sources": sources}


class CoordinatorAgent:
    """Agent to coordinate the workflow between other agents."""

    def __init__(self):
        self.doc_processor = DocumentProcessorAgent()
        self.indexer = IndexerAgent()
        self.retriever = RetrieverAgent()
        self.answer_generator = AnswerGeneratorAgent()
        self.chunks = None

    def process_query(self, pdf_path: str, query: str) -> Dict:
        """Coordinate the entire query processing workflow."""
        # Step 1: Process document
        documents = self.doc_processor.load_pdf(pdf_path)
        if not documents:
            return {"error": "No text extracted from PDF. Please check the file."}

        self.chunks = self.doc_processor.chunk_documents(documents)
        if not self.chunks:
            return {"error": "No chunks created. Please check the PDF content."}

        # Step 2: Create FAISS index
        index, texts = self.indexer.create_faiss_index(self.chunks)

        # Step 3: Retrieve relevant chunks
        context_chunks = self.retriever.retrieve_context(
            index, texts, self.chunks, query
        )

        # Step 4: Generate answer
        response = self.answer_generator.generate_answer(query, context_chunks)
        return response


# Streamlit UI
def main():
    st.title("RAG QA System for TechTrend Policies with AI Agents")
    st.markdown(
        "Ask questions about TechTrend Innovations policies based on the provided dataset."
    )

    # Initialize coordinator agent
    coordinator = CoordinatorAgent()

    # Query input
    query = st.text_input(
        "Enter your question:", value="How many days of annual leave are provided?"
    )

    # Process button
    if st.button("Get Answer") and query:
        with st.spinner("Processing PDF and generating answer..."):
            try:
                # Process query through coordinator
                response = coordinator.process_query(PDF_FILE, query)

                if "error" in response:
                    st.error(response["error"])
                    return

                # Display results
                st.subheader("Answer")
                st.write(response["answer"])

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
