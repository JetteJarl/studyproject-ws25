"""Streamlit RAG UI.

Lightweight Streamlit UI that wires an embedding model, a local
vectorstore retriever and a configurable LLM. 
"""

import streamlit as st
from langchain_core.vectorstores import VectorStoreRetriever

from document_loader import load_web_page
from document_splitter import split_documents
from embed_model import get_embeddings_model
from vector_database import save_vectorstore, load_vectorstore, get_retriever
from llm_model import MistralModel, LangchainOllamaModel, LlmModel
from user_interface import init_page

from langchain_ollama.chat_models import ChatOllama
import requests
import json

OLLAMA_URL = "http://localhost:11434"


def extract_model_name(tag: str) -> str:
    # take everything before the first ':' (or the whole string if no colon)
    return tag.split(":", 1)[0]

def get_ollama_models() -> list:
    thelist = requests.get(OLLAMA_URL+"/api/tags")
    jsondata = thelist.json()
    result = list()

    for model in jsondata["models"]:
        result.append(model["model"])

    return result


def load_system(
    model_name: str,
    all_llms: dict,
    embedder: str,
    n_chunks: int
) -> tuple[VectorStoreRetriever, LlmModel]:
    """
    Load the vectorstore, retriever, and llm based on the selected configuration

    Args:
        model_name: name of the selected llm,
        all_llms: dict with all available modes and their associated client,
        embedder: embedding model
        n_chunks: number of relevant chunks

    Returns:
        Tuple(retriever, llm)
    """
    # Load LLM
    client = all_llms[model_name]     # select corresponding client
    llm = client(model_name)      # load llm

    embeddings_model, embedder  = get_embeddings_model(embedder)
    datastore_name = "chroma_db"

    # Attempt to load an existing vectorstore; if not found, ingest from the URL
    vectorstore = load_vectorstore(embeddings_model, datastore_name)
    if vectorstore is None:
        st.error("There is no data store available. Look at the .README for more insights or contact the GitHub contributors.")

    # Configure retriever (k controls number of top documents to fetch)
    retriever = get_retriever(vectorstore, n_chunks)

    return retriever, llm



def main() -> None:
    """Streamlit app entry point.

    Sets up session-state defaults, renders the About/Settings/Input UI,
    and manages applying staged settings and storing runtime objects
    (``retriever``, ``llm``) in ``st.session_state``.
    """
    # Setting variables
    all_llms = {
        "open-mixtral-8x7b" : MistralModel,
        "mistral-small-2506": MistralModel
    }
    
    # Checking for local models
    model_list = get_ollama_models()
    print(model_list)

    for m in model_list:
        model_name = extract_model_name(m)
        all_llms.update({model_name: LangchainOllamaModel})


    default_llm = list(all_llms.keys())[0]
    default_embedder = "sentence-transformers/all-mpnet-base-v2"

    # Initialize session state defaults on first load (these are the currently applied settings)
    if "selected_llm" not in st.session_state:
        st.session_state.selected_llm = default_llm
    if "embedder" not in st.session_state:
        st.session_state.embedder = default_embedder
    if "number_relevant_chunks" not in st.session_state:
        st.session_state.number_relevant_chunks = 3
    if "retriever" not in st.session_state or "llm" not in st.session_state:
        # attempt to load initial system (will show error if no vectorstore)
        retr, lm = load_system(st.session_state.selected_llm, all_llms, st.session_state.embedder, st.session_state.number_relevant_chunks)
        st.session_state.retriever = retr
        st.session_state.llm = lm

    # Initialize temporary widget state (these reflect widget inputs before the user "applies" them)
    # Callback to mark that the user modified a temporary widget value.
    def _mark_tmp_modified():
        st.session_state.tmp_user_modified = True

    # Ensure the flag exists in session_state
    st.session_state.setdefault("tmp_user_modified", False)

    # If the user hasn't modified any tmp widgets in this session, keep the
    # tmp values in sync with the applied (selected_) values. This ensures the
    # number input displays the applied default (3) unless the user changes it.
    if not st.session_state.get("tmp_user_modified", False):
        st.session_state.selected_llm_tmp = st.session_state.selected_llm
        # coerce to int and fall back to applied value on error
        try:
            st.session_state.number_relevant_chunks_tmp = int(st.session_state.number_relevant_chunks)
        except Exception:
            st.session_state.number_relevant_chunks_tmp = st.session_state.number_relevant_chunks
        st.session_state.embedder_tmp = st.session_state.embedder

    # If settings were just applied in the previous run, keep the temporary
    # widget state in sync with the applied values before any widgets are
    # instantiated. We set `just_applied` in the Apply button handler and
    # perform the actual tmp-key assignments here (early) to avoid the
    # StreamlitAPIException that occurs when modifying a widget-bound key
    # after the widget has already been created.
    if st.session_state.get("just_applied", False):
        st.session_state.selected_llm_tmp = st.session_state.selected_llm
        st.session_state.number_relevant_chunks_tmp = st.session_state.number_relevant_chunks
        st.session_state.embedder_tmp = st.session_state.embedder
        st.session_state.just_applied = False

    # Initialize the Streamlit UI
    init_page()

    # DEBUG: show relevant session_state values to troubleshoot widget defaults
    # Remove this block after debugging
    # try:
    #     st.write({
    #         "number_relevant_chunks": st.session_state.get("number_relevant_chunks"),
    #         "number_relevant_chunks_tmp": st.session_state.get("number_relevant_chunks_tmp"),
    #         "selected_llm": st.session_state.get("selected_llm"),
    #         "selected_llm_tmp": st.session_state.get("selected_llm_tmp"),
    #     })
    # except Exception:
    #     pass

    # On first app load: show only title + start button, nothing else.
    if not st.session_state.get("rag_initialized", False):
        st.stop()
    
    with st.expander("About", expanded=True):
        st.write("This system is designed to generate convincing counterstatements to false statements made by users in the internet.\n" \
        "In recent years it has become increasingly difficult to discern false statements from fact especially in discussions but especially in social media. \n" \
        "Since our goal should not only be to label false statement as is, this site thrives to generate and present counterstatements that reach poeple")

        st.write("Our site uses a combination of AI and a manually maintained maintained database to not only be able to respond fast but also acurately to any given statement. The database consists of a combination of scientific publications, reports, and news articles from trusted sources.")

    # Settings Menu
    with st.expander("Settings"):
        small_column1, larger_column = st.columns([1, 5])
        
        with small_column1:
            # bind to temporary session_state keys so changes are not applied immediately
            number_relevant_chunks = st.number_input(
                label="Number of relevant chunks",
                min_value=1,
                max_value=20,
                step=1,
                help="How many context chunks the retriever should return for generating an answer. (Default: 3)",
                key="number_relevant_chunks_tmp",
                on_change=_mark_tmp_modified
            )

        with larger_column:
            llm_options = list(all_llms.keys())
            # Ensure tmp value is valid; if not, fall back to the first option
            if st.session_state.selected_llm_tmp not in llm_options:
                st.session_state.selected_llm_tmp = llm_options[0]
            # Use key-only mode so Streamlit reads/writes the value from session_state
            # Avoid passing `index=` together with `key=` which causes widget/session conflicts
            st.selectbox(
                "Select an LLM to be used:",
                llm_options,
                key="selected_llm_tmp",
                on_change=_mark_tmp_modified
            )

        _, button_column = st.columns([9,1])
        with button_column:
            # Determine whether any temporary settings differ from the applied ones
            settings_changed = (
                st.session_state.selected_llm_tmp != st.session_state.selected_llm
                or st.session_state.number_relevant_chunks_tmp != st.session_state.number_relevant_chunks
                or st.session_state.embedder_tmp != st.session_state.embedder
            )

            # The Apply button is disabled (greyed out) when there's no change
            if st.button("Apply Changes", disabled=not settings_changed):
                # Copy tmp -> applied
                st.session_state.selected_llm = st.session_state.selected_llm_tmp
                st.session_state.number_relevant_chunks = st.session_state.number_relevant_chunks_tmp
                st.session_state.embedder = st.session_state.embedder_tmp

                # Mark that we just applied settings; on the next rerun we will
                # copy applied -> tmp before widgets are created (see above).
                st.session_state.just_applied = True
                # Reset the tmp_user_modified flag so tmp values will be synced
                # from the applied values on the next run (clean state).
                st.session_state.tmp_user_modified = False

                # Reload system with the chosen LLM and settings, store back into session_state
                retriever, llm = load_system(
                    st.session_state.selected_llm,
                    all_llms,
                    st.session_state.embedder,
                    st.session_state.number_relevant_chunks
                )
                st.session_state.retriever = retriever
                st.session_state.llm = llm
                # No explicit rerun needed: Streamlit automatically reruns the script
                # after a widget interaction (the Apply button), so the updated
                # session_state will be picked up on the next run.


    # Display which LLM is currently applied and the number of relevant
    # chunks on a single line so they appear closer together.
    selected_llm = st.session_state.get("selected_llm")
    n_chunks = st.session_state.get("number_relevant_chunks")
    st.markdown(f"**Model:** {selected_llm} &nbsp;&nbsp; **Chunks:** {n_chunks}")

    st.markdown("**Input:**")
    query = st.text_area(
        label="Say something...",
        value="Climate change is not real. It is made up by communists to destroy the world economy.",
        help="Copy&paste a comment, a post, an entire (fake) news article etc. from social media or somewhere else."
    )

    if st.button("Generate Answer", 
                 help="Sends your input to an llm to generate a counterstatement"):
        if query:
            with st.spinner("Generating answer..."):
                # Read retriever/llm from session_state for generating answers
                retriever = st.session_state.get("retriever")
                llm = st.session_state.get("llm")
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