import streamlit as st
from langchain_core.vectorstores import VectorStoreRetriever

from document_loader import load_web_page
from document_splitter import split_documents
from embed_model import get_embeddings_model
from vector_database import save_vectorstore, load_vectorstore, get_retriever
from llm_model import MistralModel, LlmModel
from user_interface import init_page

def load_rag(
    chain: LlmModel,
    llm: str,
    embedder: str,
    number_relevant_chunks: int
) -> tuple[VectorStoreRetriever, LlmModel, str, str]:
    # TODO: Rework this method
    """
    Run the RAG pipeline for a single query and return the model answer plus the
    retrieved contexts in rank order.

    Returns:
        A tuple (answer, contexts) where:
          - answer is the generated answer as a string.
          - contexts is a list of retrieved Document objects in ranking order.
            For Ragas, convert these to a list of strings (e.g., [d.page_content for d in contexts]).
          - llm is the name of the loaded Ollama model, e.g., "llama3".
          - embedder is the name of the loaded embedding model, e.g., "sentence-transformers/all-mpnet-base-v2".
    """
    # Input: URL to scrape/load and index into the vector store if not present
    url = "https://en.wikipedia.org/wiki/COVID-19"

    # Initialize the embedding model name used for both indexing and retrieval
    embeddings_model, embedder  = get_embeddings_model(embedder)
    datastore_name = "chroma_db"

    # Attempt to load an existing vectorstore; if not found, ingest from the URL
    vectorstore = load_vectorstore(embeddings_model, datastore_name)
    if vectorstore is None:
        # Load and split documents before storing
        docs = load_web_page(url)
        # Chunking helps the retrieval quality and token efficiency
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
        vectorstore = save_vectorstore(chunks, embeddings_model, datastore_name)

    # Configure retriever (k controls number of top documents to fetch)
    retriever = get_retriever(vectorstore, number_relevant_chunks)
    return retriever, chain, llm, embedder

def main() -> None:
    """
    Entry point for the Streamlit RAG prototype.

    Steps:
    - Read URL input.
    - Initialize embeddings and vectorstore (load existing or build from URL).
    - Build retriever and LLM chain.
    - Accept a user query and generate an answer using retrieved context.
    """
    # Setting variables
    all_llms = {
        "open-mixtral-8x7b" : MistralModel,
        "mistral-small-2506": MistralModel
    }

    selected_llm = "open-mixtral-8x7b"
    default_embedder = "sentence-transformers/all-mpnet-base-v2"

    # Initialize the Streamlit UI
    init_page()

    # On first app load: show only title + start button, nothing else.
    # After clicking "Start": init_page() should set st.session_state["rag_initialized"] = True.
    if not st.session_state.get("rag_initialized", False):
        st.stop()
    
    with st.expander("About", expanded=True):
        st.write("This system is designed to generate convincing counterstatements to false statements made by users in the internet.\n" \
        "In recent years it has become increasingly difficult to discern false statements from fact especially in discussions but especially in social media. \n" \
        "Since our goal should not only be to label false statement as is, this site thrives to generate and present counterstatements that reach poeple")

        st.write("Our site uses a combination of AI and a manually maintained maintained database to not only be able to respond fast but also acurately to any given statement. The database consists of a combination of scientific publications, reports, and news articles from trusted sources.")

        st.write(f"The system can be configured to use different llms or embeeding models. Currently it is using the {selected_llm} as llm and the {default_embedder} as embedding model.")

    # Settings Menu
    with st.expander("Settings"):
        # TODO: Add embedding selection
        # Reduce the width of the number input field to make it more compact
        small_column, larger_column = st.columns([1, 5])  # 10% of container width /

        
        with small_column:
            # UI control to choose the number of relevant chunks (top-k)
            number_relevant_chunks = st.number_input(
                label="Number of relevant chunks",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                help="How many context chunks the retriever should return for generating an answer. (Default: 3)"
            )

        with larger_column:
            llm_options = all_llms.keys()
            selected_llm = st.selectbox("Select an LLM to be used:", llm_options, index=0)


    # Instantiate LLM
    # TODO: Check if selected llm is updated in reselection (are texts updated, maybe add a message) updates
    client = all_llms[selected_llm]     # select corresponding client
    llm = client(selected_llm)      # load llm

    query = st.text_area(
        label="Say something:",
        value="Climate change is not real. It is made up by communists to destroy the world economy.",
        help="Copy&paste a comment, a post, an entire (fake) news article etc. from social media or somewhere else."
    )

    retriever = load_rag(client, llm, default_embedder, number_relevant_chunks)[0]  # access retriever

    # Generate and display an answer and the retrieved context
    system_help_string = f"Your input is being processed by a RAG system. We are using a manually compiled database of sources to check claims being made in your input.\n Model: {llm}"
    if st.button("Generate Answer", 
                 help=system_help_string):
        if query:
            # Show loading circle
            with st.spinner("Generating answer..."):
                answer, context = llm.generate_answer(query, retriever)
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