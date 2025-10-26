from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

def build_vectorstore(
    chunks: List[Document],
    embeddings_model: HuggingFaceEmbeddings,
    persist_directory: Optional[str] = None,
) -> Chroma:
    if persist_directory:
        database = Chroma.from_documents(chunks, embeddings_model, persist_directory=persist_directory)
        return database
    else:
        database = Chroma.from_documents(chunks, embeddings_model)
        return database

def get_retriever(vectorstore: Chroma, k: int = 3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

# there are many more vectorstores provided by langchain
# like e.g. FAISS, Qdrant, etc.