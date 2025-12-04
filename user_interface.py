import streamlit as st

def init_page() -> None:
    """
    Configure and render the Streamlit page header.
    """
    st.set_page_config(
        page_title="RAG Pipeline Prototype",
        page_icon="🤖",
        layout="wide",
    )
    st.title("Automated Counterstatement Generation against Misinformation via Generative AI")

    # Track initialization and one-time success banner
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False
    if "show_success_once" not in st.session_state:
        st.session_state.show_success_once = False

    # Show the button only if not yet initialized
    if not st.session_state.rag_initialized:
        if st.button("Start RAG System"):
            st.session_state.rag_initialized = True
            st.session_state.show_success_once = True
            st.rerun()

    # Only run the RAG pipeline if initialized
    if st.session_state.rag_initialized and st.session_state.show_success_once:
            st.success("RAG Pipeline initialized!")
            st.session_state.show_success_once = False