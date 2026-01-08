from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)

from document_retriever import retrieve_docs

from mistralai import Mistral


import abc
import os

from dotenv import load_dotenv




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
            query: str,
            retriever,
        ) -> tuple[str, list[Document]]:
        """
        Generate an answer grounded in the retrieved context.

        Args:
            query: The user query.
            retriever: A retriever object.

        Returns:
            Model answer as a string and the retrieved documents.
        """
        pass



class langchain_model(llm_model):
    def __init__(self, model: str):
        """
        Creates langchain model class and invokes build_llm.

        Args:
            model: mdoel identifier
        """

        self.chain = self.build_llm(model)

        # there are many more models provided by langchain
        # like e.g. llama2, llama3, gpt-4, etc.


    def build_llm(self, model: str) -> Runnable:
        """
        Returns chain used as model by langchain
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
    ) -> tuple[str, list[Document]]:
        """
        Generate an answer grounded in the retrieved context.

        Args:
            query: The user query.
            retriever: A retriever object.
            chain: Runnable chain created by build_llm.

        Returns:
            Model answer as a string and the retrieved documents.
        """
        # Retrieve top documents for the question
        docs = retrieve_docs(retriever, query)
        context = _format_context(docs)
        print("Generating answer...")
        # The chain returns a message-like object with .content
        answer = (self.chain.invoke({"query": query, "context": context})).content
        return answer, docs

    # there are many more models provided by langchain
    # like e.g. llama2, llama3, gpt-4, etc.



class mistral_model(llm_model):
    def __init__(self, model: str):
        """
        Creates langchain model class and invokes build_llm.

        Args:
            model: mdoel identifier
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
        docs = retrieve_docs(query, retriever)
        context = _format_context(docs)
    
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

        return chat_response.choices[0].message.content, docs