from pathlib import Path
from typing import List, Optional, Dict
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from document_loader import load_web_page
from document_splitter import split_documents


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

def add_url(
    vectorstore: Optional[VectorStore],
    url: str,
    embeddings_model: Optional[HuggingFaceEmbeddings] = None,
    persist_directory: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> VectorStore:
    """ 
    Reads a single URL into the given vectorstore.
    """

    # Load and split the page
    docs = load_web_page(url)
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # If there is no existing store, create one (requires embeddings and persist dir)
    if vectorstore is None:
        if embeddings_model is None or persist_directory is None:
            raise ValueError("embeddings_model and persist_directory required to create a new vectorstore")
        return save_vectorstore(chunks, embeddings_model, persist_directory)

    # Add documents
    if hasattr(vectorstore, "add_documents"):
        print(f"Adding {len(chunks)} chunks from {url} to existing vectorstore...")
        vectorstore.add_documents(chunks)
        # persist if supported
        if hasattr(vectorstore, "persist"):
            try:
                vectorstore.persist()
            except Exception:
                pass
        return vectorstore

    # Raise Error if failed
    raise RuntimeError("Vectorstore does not support adding documents and cannot create a new one")
 

def get_retriever(vectorstore: VectorStore, number_relevant_chunks: int) -> VectorStoreRetriever:
    """
    Create a retriever from the vector store.

    Args:
        vectorstore: The underlying vector database.
        number_relevant_chunks: Number of top documents to retrieve.

    Returns:
        A retriever configured to return top-k matches.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": number_relevant_chunks})
    return retriever

# there are many more vectorstores provided by langchain
# like e.g. FAISS, Qdrant, etc.


def list_current_database(vectorstore: Optional[VectorStore]) -> List[Dict[str, str]]:
    """
    Return a list of dicts describing the current database contents.

    The returned list contains dicts with keys: 'title', 'source'.

    This function performs only data access / parsing and does not depend on
    Streamlit or any UI code. It is intentionally defensive and will return
    an empty list if the store is missing or cannot be parsed.

    Args:
        vectorstore: The loaded VectorStore (or None).

    Returns:
        A list of dictionaries with keys 'title','source'.
    """
    # Defensive checks
    if vectorstore is None:
        return []

    # Try to obtain a serializable snapshot of stored documents.
    data = None
    try:
        if hasattr(vectorstore, "get"):
            data = vectorstore.get()
        elif hasattr(vectorstore, "_collection") and hasattr(vectorstore._collection, "get"):
            data = vectorstore._collection.get()
        else:
            # last resort: try introspecting common attributes
            for candidate in ("client", "collection", "_client", "_store"):
                obj = getattr(vectorstore, candidate, None)
                if obj is not None and hasattr(obj, "get"):
                    data = obj.get()
                    break
    except Exception:
        data = None

    rows: List[Dict[str, str]] = []
    try:
        # Chroma/LangChain shape: dict with 'metadatas' and 'documents'
        if isinstance(data, dict):
            metadatas = data.get("metadatas") or data.get("metadata") or []
            documents = data.get("documents") or []
            # If metadatas present, prefer them
            if metadatas:
                for meta in metadatas:
                    title = meta.get("title") or meta.get("name") or ""
                    source = meta.get("source") or meta.get("url") or meta.get("link") or ""
                    rows.append({"title": title, "source": source})
            elif documents:
                # documents may be list of strings or Document objects
                for d in documents:
                    if isinstance(d, str):
                        rows.append({"type": "", "title": "", "source": d[:200]})
                    else:
                        meta = getattr(d, "metadata", {}) or {}
                        title = meta.get("title") or ""
                        source = meta.get("source") or ""
                        rows.append({"title": title, "source": source})
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    title = item.get("title") or ""
                    source = item.get("source") or item.get("url") or ""
                    rows.append({"title": title, "source": source})
                else:
                    meta = getattr(item, "metadata", {}) or {}
                    rows.append({"title": meta.get("title", ""), "source": meta.get("source", "")})
        else:
            # Try to iterate over a fallback attribute that may hold Documents
            potential = getattr(vectorstore, "documents", None) or getattr(vectorstore, "_documents", None)
            if potential:
                for d in potential:
                    meta = getattr(d, "metadata", {}) or {}
                    rows.append({"title": meta.get("title", ""), "source": meta.get("source", "")})
    except Exception:
        rows = []

    # Deduplicate rows while preserving order. 
    deduped: List[Dict[str, str]] = []
    seen = set()
    for r in rows:
        key = (r.get("title", ""), r.get("source", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    return deduped