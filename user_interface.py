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