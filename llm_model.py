from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)

from document_retriever import retrieve_docs

def _format_context(docs: list[Document]) -> str:
    """
    Convert a list of Documents into a single context string.

    Args:
        docs: Retrieved documents to be fed to the LLM.

    Returns:
        Aggregated string containing the documents' page content.
    """
    context = "\n\n".join(d.page_content for d in docs)
    return context

def build_llm(model: str) -> tuple[Runnable, str]:
    """
    Create the local Ollama-backed chat model and a simple prompt chain.

    Args:
        model: Name of the local Ollama model to load (e.g., "llama3").

    Returns:
        A tuple (chain, llm) where:
          - chain: a Runnable that accepts a dict {"query": str, "context": str}
                   and can be invoked via .invoke({...}). It produces a message-like
                   object with a .content string used as the model’s answer.
          - llm: the name of the loaded Ollama model, e.g., "llama3.
    """
    print("Loading local model...")
    llm = ChatOllama(model=model)
    prompt = ChatPromptTemplate([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant. Answer by using the provided context."
            "If you are unsure of the answer, just say that you don't know and don't make up an answer."),
        HumanMessagePromptTemplate.from_template("Query: {query}\n\nContext:\n{context}"),
    ])
    chain = prompt | llm
    return chain, model

def generate_answer(
    query: str,
    retriever,
    chain,
) -> tuple[str, list[Document]]:
    """
    Generate an answer grounded in the retrieved context.

    Args:
        query: The user query.
        retriever: A retriever object.
        chain: Runnable chain created by build_llm.

    Returns:
        Model answer as a string and the retrieved documents as a list of Document objects.
    """
    # Retrieve top documents for the question
    docs = retrieve_docs(retriever, query)
    context = _format_context(docs)
    print("Generating answer...")
    # The chain returns a message-like object with .content
    answer = (chain.invoke({"query": query, "context": context})).content
    return answer, docs

# there are many more models provided by langchain
# like e.g. llama2, llama3, gpt-4, etc.