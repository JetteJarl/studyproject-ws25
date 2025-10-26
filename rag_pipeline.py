from langchain_core.documents import Document

from document_loader import load_web_page
from document_splitter import split_documents
from document_embedding import get_embeddings_model
from vector_database import save_vectorstore, load_vectorstore, get_retriever
from llm_model import build_llm, generate_answer

def build_rag_pipeline(
    *,
    source: str,
    embed_model_name: str,
    persist_directory: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    retriever_k: int = 3
):
    embeddings_model = get_embeddings_model(embed_model_name)

    vectorstore = load_vectorstore(embeddings_model, persist_directory)
    if vectorstore is None:
        docs = load_web_page(source)
        chunks: list[Document] = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vectorstore = save_vectorstore(chunks, embeddings_model, persist_directory)

    retriever = get_retriever(vectorstore, k=retriever_k)
    return retriever

if __name__ == "__main__":
    retriever = build_rag_pipeline(
        source="https://en.wikipedia.org/wiki/COVID-19",
        chunk_size=1000,
        chunk_overlap=200,
        embed_model_name="sentence-transformers/all-mpnet-base-v2",
        retriever_k=3,
        persist_directory="chroma_db",
    )

    chain = build_llm(model="llama3")
    question = "Is the vaccine effective?"
    answer = generate_answer(question=question, retriever=retriever, chain=chain)
    print(answer)