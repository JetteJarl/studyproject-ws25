import streamlit as st

from document_loader import load_web_page
from document_splitter import split_documents
from embed_model import get_embeddings_model
from vector_database import save_vectorstore, load_vectorstore, get_retriever
from llm_model import build_llm, generate_answer
from user_interface import init_page

def main() -> None:
    """
    Entry point for the Streamlit RAG prototype.

    Steps:
    - Read URL input.
    - Initialize embeddings and vectorstore.
    - Build retriever and LLM chain.
    - Accept a user query and generate an answer using retrieved context.
    """
    # Initialize the Streamlit UI
    init_page()

    # Only run the RAG pipeline and show query UI if initialized
    if st.session_state.get("rag_initialized", False):
        # Initialize the embedding model name used for both indexing and retrieval
        embeddings_model = get_embeddings_model("sentence-transformers/all-mpnet-base-v2")
        datastore_name = "chroma_db"

        # Attempt to load an existing vectorstore; if not found, ingest from the URL
        vectorstore = load_vectorstore(embeddings_model, datastore_name)
        if vectorstore is None:
            # Load and split documents before storing
            docs = load_web_page("https://en.wikipedia.org/wiki/COVID-19")
            # Chunking helps the retrieval quality and token efficiency
            chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
            vectorstore = save_vectorstore(chunks, embeddings_model, datastore_name)

        # Configure retriever (k controls number of top documents to fetch)
        retriever = get_retriever(vectorstore, k=3)

        # Build the LLM chain and accept a user query
        chain = build_llm("tinyllama")
        query = st.text_area(label="Say something: ", value="Is the vaccine effective?")

        # Generate and display an answer and the retrieved context
        if st.button("Get Answer"):
            if query:
                # Show loading circle
                with st.spinner("Generating answer..."):
                    answer, context = generate_answer(query, retriever, chain)
                    st.write("Answer:")
                    st.write(answer)
                    with st.expander("Show Retrieved Context"):
                        for i, doc in enumerate(context, 1):
                            st.markdown(f"**Relevant Chunk {i}:**")
                            st.markdown(doc.page_content)
                            st.markdown(f"*Source:* {doc.metadata['source']}")
                            st.markdown("---")
            else:
                st.warning("Please enter some text.")

if __name__ == "__main__":
    main()