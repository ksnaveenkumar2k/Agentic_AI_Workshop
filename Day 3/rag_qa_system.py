import os
import streamlit as st
import fitz 
import faiss
import numpy as np
import google.generativeai as genai
import re
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Streamlit page configuration
st.set_page_config(page_title="RAG QA System", layout="wide")

# Define the database path for PDFs
DATABASE_PATH = r"D:/ihub_task_agentic_ai/Agentic_AI_Workshop/Day 3/RAG Project Dataset"
PDF_FILES = [
    os.path.join(DATABASE_PATH, "1706.03762v7.pdf"),
    os.path.join(DATABASE_PATH, "2005.11401v4.pdf"),
    os.path.join(DATABASE_PATH, "2005.14165v4.pdf"),
    os.path.join(DATABASE_PATH, "SNS.pdf")
]


# Configure Gemini API (replace with your API key)
genai.configure(api_key="AIzaSyD5ZaF3CkKdqTsz8kZD1HZWPtTpuoXPTfw")

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

model = load_embedding_model()

# Step 1: Document Preprocessing
def load_pdfs(pdf_paths: List[str]) -> List[Dict]:
    """Load and extract text from PDF files in the database."""
    documents = []
    for pdf_path in pdf_paths:
        try:
            with open(pdf_path, 'rb') as file:
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

def chunk_documents(documents: List[Dict], chunk_size=5) -> List[Dict]:
    """Chunk documents by sentences."""
    chunks = []
    for doc in documents:
        text = doc["text"]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for i in range(0, len(sentences), chunk_size):
            chunk_text = " ".join(sentences[i:i+chunk_size])
            if len(chunk_text.strip()) > 0:
                chunks.append({
                    "text": chunk_text,
                    "source": doc["source"],
                    "chunk_id": f"{doc['source']}_chunk_{i // chunk_size}"
                })
    return chunks

# Step 2: Embedding and FAISS Index
@st.cache_resource
def create_faiss_index(_chunks: List[Dict]):
    """Create a FAISS index from document chunks."""
    texts = [chunk["text"] for chunk in _chunks]
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, texts

# Step 3: Retrieval
def retrieve_context(index, texts: List[str], query: str, k: int = 5) -> List[Dict]:
    """Retrieve relevant chunks based on the query."""
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    return [
        {
            "text": texts[i],
            "source": _chunks[i]["source"],
            "chunk_id": _chunks[i]["chunk_id"]
        }
        for i in indices[0]
    ]

# Step 4: Answer Generation
def generate_answer(query: str, context_chunks: List[Dict]) -> Dict:
    """Generate an answer using Gemini API."""
    context = "\n\n".join([f"Chunk {i+1}:\n{chunk['text']}" for i, chunk in enumerate(context_chunks)])
    prompt = (
        "You are an expert AI researcher. Based only on the following context from research papers, "
        "provide a detailed, accurate, and well-cited answer to the question. If relevant, mention equations, "
        "figures, or technical details from the text.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"
    
    sources = [
        {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"][:300] + "..."
        }
        for chunk in context_chunks
    ]
    return {"answer": answer, "sources": sources}

# Streamlit UI
def main():
    st.title("RAG QA System for AI Research Papers")
    st.markdown("Ask questions about transformer models based on preloaded research papers.")

    # Query input
    query = st.text_input(
        "Enter your question:",
        value="What are the main components of a RAG model, and how do they interact?"
    )

    # Process button
    if st.button("Get Answer") and query:
        with st.spinner("Processing PDFs and generating answer..."):
            try:
                # Load and process PDFs from the database
                documents = load_pdfs(PDF_FILES)
                if not documents:
                    st.error("No text extracted from PDFs. Please check the files in the database.")
                    return

                global _chunks
                _chunks = chunk_documents(documents)
                if not _chunks:
                    st.error("No chunks created. Please check the PDF content.")
                    return

                # Create FAISS index
                index, texts = create_faiss_index(_chunks)

                # Retrieve relevant chunks
                context_chunks = retrieve_context(index, texts, query, k=5)

                # Generate answer
                response = generate_answer(query, context_chunks)

                # Display results
                st.subheader("Answer")
                st.write(response["answer"])

                st.subheader("Sources")
                for source in response["sources"]:
                    with st.expander(f"{source['source']} (Chunk: {source['chunk_id']})"):
                        st.write(source["text"])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
