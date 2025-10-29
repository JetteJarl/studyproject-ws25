from pathlib import Path
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

def save_vectorstore(
    chunks: List[Document],
    embeddings_model: HuggingFaceEmbeddings,
    persist_directory: str,
) -> VectorStore:
    """
    Build and persist a Chroma vector store from document chunks.

    Args:
        chunks: Pre-split documents used for indexing.
        embeddings_model: Embeddings model used for vectorization.
        persist_directory: Directory/name for the Chroma collection.

    Returns:
        A persisted VectorStore instance.
    """
    # Create a Chroma collection from chunks.
    print("Building and saving new database...")
    database = Chroma.from_documents(chunks, embeddings_model, persist_directory=persist_directory)
    return database

def load_vectorstore(
    embeddings_model: HuggingFaceEmbeddings,
    persist_directory: str,
) -> Optional[VectorStore]:
    """
    Load an existing Chroma vector store if available.

    Args:
        embeddings_model: Embeddings model used at indexing time (must match).
        persist_directory: Directory/name for the Chroma collection.

    Returns:
        The loaded VectorStore, or None if the collection doesn't exist.
    """
    path = Path(persist_directory)
    has_data = path.exists() and any(path.iterdir())
    if has_data:
        # Load existing database
        print("Loading existing database...")
        database = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
        return database
    return None

def get_retriever(vectorstore: VectorStore, k: int = 3) -> VectorStoreRetriever:
    """
    Create a retriever from the vector store.

    Args:
        vectorstore: The underlying vector database.
        k: Number of top documents to retrieve.

    Returns:
        A retriever configured to return top-k matches.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

# there are many more vectorstores provided by langchain
# like e.g. FAISS, Qdrant, etc.