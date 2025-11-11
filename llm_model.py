from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)

from document_retriever import retrieve_docs

import abc


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

class llm_model(abc.ABC):
    @abc.abstractmethod
    def build_llm(self, model: str) -> Runnable:
        """
        Construct a simple LLM chain with a system and human prompt.

        Args:
            model: Name of the local Ollama model (e.g., 'llama3').

        Returns:
            A runnable chain compatible with .invoke({"query": ..., "context": ...}).
        """
        pass
    
    @abc.abstractmethod
    def generate_answer(
            self,
            query: str,
            retriever,
        ) -> str:
        """
        Generate an answer grounded in the retrieved context.

        Args:
            query: The user query.
            retriever: A retriever object.

        Returns:
            Model answer as a string.
        """
        pass



class olama_model(llm_model):
    def __init__(self, model: str):
        super().__init__()
        self.chain = self.build_llm(model)

        # there are many more models provided by langchain
        # like e.g. llama2, llama3, gpt-4, etc.


    def build_llm(self, model: str) -> Runnable:
        """
        Construct a simple LLM chain with a system and human prompt.

        Args:
            model: Name of the local Ollama model (e.g., 'llama3').

        Returns:
            A runnable chain compatible with .invoke({"query": ..., "context": ...}).
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
        return chain


    def generate_answer(
        self,
        query: str,
        retriever,
    ) -> str:
        """
        Generate an answer grounded in the retrieved context.

        Args:
            query: The user query.
            retriever: A retriever object.
            chain: Runnable chain created by build_llm.

        Returns:
            Model answer as a string.
        """
        # Retrieve top documents for the question
        docs = retrieve_docs(retriever, query)
        context = _format_context(docs)
        print("Generating answer...")
        # The chain returns a message-like object with .content
        answer = (self.chain.invoke({"query": query, "context": context})).content
        return answer


