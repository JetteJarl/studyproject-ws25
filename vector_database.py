from pathlib import Path
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

def save_vectorstore(
    chunks: List[Document],
    embeddings_model: HuggingFaceEmbeddings,
    persist_directory: str,
) -> Chroma:
    # Build and persist database
    print("Building and saving new database...")
    database = Chroma.from_documents(chunks, embeddings_model, persist_directory=persist_directory)
    return database

def load_vectorstore(
    embeddings_model: HuggingFaceEmbeddings,
    persist_directory: str,
) -> Chroma | None:
    path = Path(persist_directory)
    has_data = path.exists() and any(path.iterdir())
    if has_data:
        # Load existing database
        print("Loading existing database...")
        database = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
        return database
    return None

def get_retriever(vectorstore: Chroma, k: int = 3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

# there are many more vectorstores provided by langchain
# like e.g. FAISS, Qdrant, etc.