import abc
import os
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from mistralai import Mistral
from dotenv import load_dotenv

from document_retriever import retrieve_docs

# SYSTEM_PROMPT = "You are an expert, checking facts in statements made in public. If you are unsure do not make up information, instead say that you are missing information."

# SYSTEM_PROMPT = """You are an expert for checking facts in statements made in public. Your goal is to convince
# people without criticizing them. Keep the tone respectful and be concise.
# Here are things that make a counter statement good and helpful:
# 1. Include sources for your claims, you can for example name the entity that published data
# that you used
# 2. Use easy to understand language and formulations.
# 3. Directly address the false claims that you argue against.
# 4. Provide context.
# When you are missing information about the statement being made clearly state that you
# are missing the necessary context information to check the statement."""

SYSTEM_PROMPT = """You are an expert at communicating with people. Your goal is to convince people who believe false statements made in public of 
facts and evidence with a scientific basis. To do so you need to generate a short and concise counterstatement that can convince people.

When adressing people and confronting them with new information use your knowledge on how to communicate with people. 
Additionally use these guidelines on how to formulate counterstatements:
1. Include sources for your claims, you can for example name the entity that published data
that you used
2. Use easy to understand language and formulations.
3. Directly address the false claims that you argue against.
4. Provide context to the arguments you use

Any statements you make need to be based on evidence. You may only use the information provided as context together with the user query. 
If you are missing information to respond to the false statement state so cleary. Do not make up any information for the purpose of responding.
"""

# SYSTEM_PROMPT = """You are an expert at communicating with people. Your goal is to convince people who believe false statements made in public of 
# facts and evidence with a scientific basis. To do so you need to generate a counterstatement and a rating on a scale 0-2 where 0 is a statement that is completly true,
# 1 is a statement that is related to evidence but misinterprets it or takes it out of context, 2 is a statement that has no ties to facts or evidence.

# When adressing people and confronting them with new information use your knowledge on how to communicate with people. 
# Additionally use these guidelines on how to formulate counterstatements:
# 1. Include sources for your claims, you can for example name the entity that published data
# that you used
# 2. Use easy to understand language and formulations.
# 3. Directly address the false claims that you argue against.
# 4. Provide context.

# Any statements you make need to be based on evidence. You may only use the information provided as context together with the user query. 
# If you are missing information to respond to the false statement state so cleary. Do not make up any information for the purpose of responding.

# Your counterstatement should fit the template:
# <your generated counter arguments to the statement>
# Rating: <0/1/2>
# """

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
        raise NotImplementedError
    
    @abc.abstractmethod
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

        Returns:
            Model answer as a string along with the retrieved context.
        """
        raise NotImplementedError



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
        docs = retrieve_docs(retriever, query)
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
                "content": SYSTEM_PROMPT
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
        return answer, docs