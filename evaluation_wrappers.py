import os
import asyncio
from typing import Any, List

from dotenv import load_dotenv
from langchain_core.outputs import Generation, LLMResult
from mistralai import Mistral

from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import BaseRagasEmbeddings


class _MistralAsyncAdapter:
    """
    Adapter implementing the subset of LangChain LLM interface that Ragas
    (through LangchainLLMWrapper) uses: `agenerate_prompt`.

    It returns a LangChain `LLMResult` with `Generation` objects.
    """

    def __init__(self, model: str):
        load_dotenv()
        api_key = os.environ["MISTRAL_API_KEY"]
        self._client = Mistral(api_key=api_key)
        self._model = model

    def _prompt_to_text(self, p: Any) -> str:
        if isinstance(p, str):
            return p
        # LangChain PromptValue has .to_string()
        to_string = getattr(p, "to_string", None)
        if callable(to_string):
            return to_string()
        return str(p)

    def _complete_one(self, prompt_text: str) -> str:
        resp = self._client.chat.complete(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a strict evaluator. Follow instructions exactly."},
                {"role": "user", "content": prompt_text},
            ],
        )
        return resp.choices[0].message.content

    async def agenerate_prompt(self, prompts: List[Any], **kwargs: Any) -> LLMResult:
        prompt_texts = [self._prompt_to_text(p) for p in prompts]

        async def _run_one(t: str) -> str:
            return await asyncio.to_thread(self._complete_one, t)

        outputs = await asyncio.gather(*[_run_one(t) for t in prompt_texts])

        generations = [[Generation(text=o)] for o in outputs]
        return LLMResult(generations=generations)

    async def apredict(self, prompt: str) -> str:
        # Convenience; not required if agenerate_prompt exists, but harmless.
        return await asyncio.to_thread(self._complete_one, prompt)


class MistralRagasLLM(LangchainLLMWrapper):
    """
    Ragas LLM wrapper using the Mistral API directly (no Ollama).

    Parameters:
        model: Mistral model name, e.g. "open-mixtral-8x7b"
    """

    def __init__(self, model: str):
        load_dotenv()
        self.langchain_llm = _MistralAsyncAdapter(model=model)
        self.bypass_temperature = True
        self.is_finished_parser = None
        self.run_config = None

    def set_run_config(self, config):
        self.run_config = config


class RagasHuggingFaceWrapper(BaseRagasEmbeddings):
    """
    Ragas embeddings adapter backed by LangChain's HuggingFaceEmbeddings.

    Implements the full embedding interface expected by recent Ragas versions,
    enabling both synchronous and asynchronous usage in metrics that require
    vector similarities.

    Parameters:
        model_name: Hugging Face model identifier (e.g., "sentence-transformers/all-mpnet-base-v2").

    Attributes:
        lc_embedder: Underlying LangChain HuggingFaceEmbeddings instance.
    """
    def __init__(self, model_name: str):
        self.lc_embedder = HuggingFaceEmbeddings(model_name=model_name)

    def embed_query(self, query: str):
        """
        Compute an embedding vector for a single query string.

        Parameters:
            query: Input text.

        Returns:
            A vector (list[float]) representing the query embedding.
        """
        return self.lc_embedder.embed_query(query)

    def embed_documents(self, docs: list[str]):
        """
        Compute embedding vectors for multiple documents.

        Parameters:
            docs: List of input texts.

        Returns:
            A list of vectors (list[list[float]]) for each document.
        """
        return self.lc_embedder.embed_documents(docs)

    async def aembed_query(self, query: str):
        """
        Asynchronously compute an embedding for a single query.

        Parameters:
            query: Input text.

        Returns:
            A vector (list[float]) representing the query embedding.
        """
        return self.lc_embedder.embed_query(query)

    async def aembed_documents(self, docs: list[str]):
        """
        Asynchronously compute embeddings for multiple documents.

        Parameters:
            docs: List of input texts.

        Returns:
            A list of vectors (list[list[float]]) for each document.
        """
        return self.lc_embedder.embed_documents(docs)

    async def embed_text(self, text: str):
        """
        Asynchronously compute an embedding for a single text (alias of embed_query).

        Parameters:
            text: Input text.

        Returns:
            A vector (list[float]) representing the text embedding.
        """
        return self.lc_embedder.embed_query(text)