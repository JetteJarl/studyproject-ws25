import abc
import os

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)
from mistralai import Mistral
from dotenv import load_dotenv

from document_retriever import retrieve_docs


def retrieve_context(query, retriever):
    """
    Retrieve the context for a given query, based on the retriever

    Args:
        query: User input
        retriever: Retriever object

    Returns:
        Formatted context matching the user's query
    """

    docs = retrieve_docs(retriever, query)
    context = _format_context(docs)

    return context


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

class LlmModel(abc.ABC):
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
            Model answer as a string along with the retrieved context.
        """
        pass

class MistralModel(LlmModel):
    def __init__(self, model: str):
        """
        Creates langchain model class and invokes build_llm.

        Args:
            model: model identifier
        """

        self.model = model
        self.client = self.build_llm(self.model)

    def build_llm(self, model):
        """
        Builds mistral model. Needs the API Key.
        """

        print("Setting up remote model access...")
        load_dotenv()
        api_key = os.environ["MISTRAL_API_KEY"]
        client = Mistral(api_key=api_key)

        return client

    def generate_answer(self, query, retriever):
        """
        Generating answer using mistral model.
        """

        # Retrieve top documents for the question
        context = retrieve_context(query, retriever)

        prompt = f"""
        Context information is below.
        ---------------------
        {context}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {query}
        Answer:
        """

        messages = [
            {
                "role":"system",
                "content": "You are an expert, checking facts in statements made in public. If you are unsure do not make up information, instead say that you are missing information."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        print("Sending request...")
        chat_response = self.client.chat.complete(
            model= self.model,
            messages = messages
        )

        answer = chat_response.choices[0].message.content
        return answer, context