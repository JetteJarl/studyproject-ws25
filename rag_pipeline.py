import streamlit as st

from document_loader import load_web_page
from document_splitter import split_documents
from embed_model import get_embeddings_model
from vector_database import save_vectorstore, load_vectorstore, get_retriever
from llm_model import langchain_model, mistral_model
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

    # Only run the RAG pipeline and show the query UI if initialized
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

        # Reduce the width of the number input field to make it more compact
        col_small, col_spacer = st.columns([1, 9]) # 10% of container width
        with col_small:
            # UI control to choose the number of relevant chunks (top-k)
            number_relevant_chunks = st.number_input(
                label="Number of relevant chunks",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                help="How many context chunks the retriever should return for generating an answer. (Default: 3)"
            )

        # Configure retriever (number_relevant_chunks controls the number of top documents to fetch)
        retriever = get_retriever(vectorstore, number_relevant_chunks)

        # Build the LLM chain and accept a user query
        mixtral = mistral_model("open-mixtral-8x7b")
        # llama3 = olama_model("llama3")

        query = st.text_area(
            label="Say something:",
            value="Is the vaccine effective?",
            help="Copy&paste a comment, a post, an entire (fake) news article etc. from social media or somewhere else."
        )

        # Generate and display an answer and the retrieved context
        if st.button("Generate Answer"):
            if query:
                # Show loading circle
                with st.spinner("Generating answer..."):
                    answer, context = mixtral.generate_answer(query, retriever)
                    st.write("Answer:")
                    st.write(answer)
                    with st.expander("Show Retrieved Context from Local Vector Database"):
                        for i, doc in enumerate(context, 1):
                            st.markdown(f"**Relevant Chunk {i}:**")
                            st.markdown(doc.page_content)
                            st.markdown(f"*Source:* {doc.metadata['source']}")
                            st.markdown("---")
            else:
                st.warning("Please enter some text.")

if __name__ == "__main__":
    main()