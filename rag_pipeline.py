from langchain_core.documents import Document
import streamlit as st

from document_loader import load_web_page
from document_splitter import split_documents
from embed_model import get_embeddings_model
from vector_database import save_vectorstore, load_vectorstore, get_retriever
from llm_model import build_llm, generate_answer

st.set_page_config(page_title="RAG Pipeline Prototype", page_icon="🤖", layout="wide")
st.title("Automated Counterstatement Generation against Misinformation via Generative AI")

url = st.text_input(label="Enter a URL to load:", value="https://en.wikipedia.org/wiki/COVID-19")

embeddings_model = get_embeddings_model("sentence-transformers/all-mpnet-base-v2")
datastore_name = "chroma_db"

vectorstore = load_vectorstore(embeddings_model, datastore_name)
if vectorstore is None:
    docs = load_web_page(url)
    chunks: list[Document] = split_documents(docs, chunk_size=1000, chunk_overlap=200)
    vectorstore = save_vectorstore(chunks, embeddings_model, datastore_name)

retriever = get_retriever(vectorstore, k=3)
st.success("RAG Pipeline initialized!")

chain = build_llm(model="llama3")
query = st.text_input(label="Say something: ", value="Is the vaccine effective?")
answer = generate_answer(question=query, retriever=retriever, chain=chain)
st.write("Answer: ", answer)