import streamlit as st

def init_page() -> None:
    """
    Configure and render the Streamlit page header.

    Behavior:
    - Always renders the title.
    - If the RAG system is not initialized yet, renders only the "Start" button
      and stops execution so the rest of the app UI doesn't appear.
    - After initialization, shows a one-time success banner and allows the app to continue.
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

    # If not initialized: show only the Start button and stop here
    if not st.session_state.rag_initialized:
        if st.button("Start RAG System"):
            st.session_state.rag_initialized = True
            st.session_state.show_success_once = True
            st.rerun()
        st.stop()

    # Initialized: show success banner once, then allow the rest of the app to render
    if st.session_state.rag_initialized and st.session_state.show_success_once:
        st.success("RAG System initialized!")
        st.session_state.show_success_once = False