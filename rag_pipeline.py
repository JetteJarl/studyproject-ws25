import streamlit as st
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from document_loader import load_web_page
from document_splitter import split_documents
from embed_model import get_embeddings_model
from vector_database import save_vectorstore, load_vectorstore, get_retriever
from llm_model import build_llm, generate_answer

def _init_page() -> None:
    """
    Configure and render the Streamlit page header.
    """
    st.set_page_config(
        page_title="RAG Pipeline Prototype",
        page_icon="🤖",
        layout="wide",
    )
    st.title("Automated Counterstatement Generation against Misinformation via Generative AI")

def run_rag(query: str, vectorstore: VectorStore) -> tuple[str, list[Document]]:
    """
    Run the RAG pipeline for a single query and return the model answer plus the
    retrieved contexts in rank order.

    Args:
        query: The user claim/s to answer.
        vectorstore: The vector store backing the retriever.

    Returns:
        A tuple (answer, contexts) where:
          - answer is the generated answer as a string.
          - contexts is a list of retrieved Document objects in ranking order.
            For Ragas, convert these to a list of strings (e.g., [d.page_content for d in contexts]).
    """
    # Configure retriever (k controls number of top documents to fetch)
    retriever = get_retriever(vectorstore, k=3)

    # Build the LLM chain and accept a user query
    chain = build_llm("llama3")

    # Generate and display an answer grounded in the retrieved context
    answer, docs = generate_answer(query, retriever, chain)
    return answer, docs

def main() -> None:
    """
    Entry point for the Streamlit RAG prototype.

    Steps:
    - Read URL input.
    - Initialize embeddings and vectorstore (load existing or build from URL).
    - Build retriever and LLM chain.
    - Accept a user query and generate an answer using retrieved context.
    """
    _init_page()

    # Input: URL to scrape/load and index into the vector store if not present
    url = st.text_input(label="Enter a URL to load:", value="https://en.wikipedia.org/wiki/COVID-19")

    # Initialize the embedding model name used for both indexing and retrieval
    embeddings_model = get_embeddings_model("sentence-transformers/all-mpnet-base-v2")
    datastore_name = "chroma_db"

    # Attempt to load an existing vectorstore; if not found, ingest from the URL
    vectorstore = load_vectorstore(embeddings_model, datastore_name)
    if vectorstore is None:
        # Load and split documents before storing
        docs = load_web_page(url)
        # Chunking helps the retrieval quality and token efficiency
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
        vectorstore = save_vectorstore(chunks, embeddings_model, datastore_name)

    query = st.text_input(label="Say something: ", value="Is the vaccine effective?")
    answer, _ = run_rag(query, vectorstore)
    st.write("Answer: ", answer)

if __name__ == "__main__":
    main()