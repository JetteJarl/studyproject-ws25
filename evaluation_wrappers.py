import os
import asyncio
import random
from typing import Any, List

import numpy as np
from dotenv import load_dotenv
from langchain_core.outputs import Generation, LLMResult
from mistralai import Mistral

from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import BaseRagasEmbeddings


class _MistralAsyncAdapter:
    """
    Adapter implementing the subset of LangChain LLM interface that Ragas uses.

    Adds:
      - concurrency limiting (semaphore)
      - retry w/ exponential backoff on HTTP 429 (rate limiting)
      - hard timeout per request to prevent "100% but never finishes"
    """

    def __init__(
        self,
        model: str,
        *,
        max_concurrency: int = 1,
        request_timeout_s: float = 90.0,
        max_retries: int = 5,
        base_backoff_s: float = 1.0,
        max_backoff_s: float = 20.0,
    ):
        load_dotenv()
        api_key = os.environ["MISTRAL_API_KEY"]
        self._client = Mistral(api_key=api_key)
        self._model = model

        self._sem = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._request_timeout_s = float(request_timeout_s)
        self._max_retries = int(max_retries)
        self._base_backoff_s = float(base_backoff_s)
        self._max_backoff_s = float(max_backoff_s)

    def _prompt_to_text(self, p: Any) -> str:
        if isinstance(p, str):
            return p
        to_string = getattr(p, "to_string", None)
        if callable(to_string):
            return to_string()
        return str(p)

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        msg = str(exc)
        return "Status 429" in msg or "rate limit" in msg.lower() or "rate_limited" in msg.lower()

    def _complete_one_sync(self, prompt_text: str) -> str:
        resp = self._client.chat.complete(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluation engine.\n"
                        "You MUST follow the user's instructions EXACTLY.\n"
                        "If the user requests JSON, output ONLY valid JSON (no markdown, no extra text).\n"
                        "Never omit required fields. If you are uncertain, still provide the required keys "
                        "with a best-effort value (e.g., verdict='unknown') and a short reason."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content

    async def _complete_one(self, prompt_text: str) -> str:
        async with self._sem:
            attempt = 0
            while True:
                try:
                    # Hard timeout around the thread call
                    return await asyncio.wait_for(
                        asyncio.to_thread(self._complete_one_sync, prompt_text),
                        timeout=self._request_timeout_s,
                    )
                except TimeoutError as exc:
                    attempt += 1
                    if attempt > self._max_retries:
                        raise
                    backoff = min(self._max_backoff_s, self._base_backoff_s * (2 ** (attempt - 1)))
                    backoff = backoff * (0.7 + 0.6 * random.random())
                    await asyncio.sleep(backoff)
                except Exception as exc:
                    attempt += 1
                    if not self._is_rate_limit_error(exc) or attempt > self._max_retries:
                        raise
                    backoff = min(self._max_backoff_s, self._base_backoff_s * (2 ** (attempt - 1)))
                    backoff = backoff * (0.7 + 0.6 * random.random())
                    await asyncio.sleep(backoff)

    async def agenerate_prompt(self, prompts: List[Any], **kwargs: Any) -> LLMResult:
        prompt_texts = [self._prompt_to_text(p) for p in prompts]
        outputs = await asyncio.gather(*[self._complete_one(t) for t in prompt_texts])
        generations = [[Generation(text=o)] for o in outputs]
        return LLMResult(generations=generations)

    async def apredict(self, prompt: str) -> str:
        return await self._complete_one(prompt)


class MistralRagasLLM(LangchainLLMWrapper):
    """
    Ragas LLM wrapper using the Mistral API directly (no Ollama).

    Parameters:
        model: Mistral model name, e.g. "open-mixtral-8x7b"
    """
    def __init__(self, model: str):
        # max_concurrency=1 is slow but stable; raise to 2 only if you stop seeing 429/TimeoutError.
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

    def _to_1d_list(self, vec) -> list[float]:
        arr = np.asarray(vec, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D embedding vector, got shape {arr.shape}")
        return arr.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Compute an embedding vector for a single query string.
        Must return a 1D list[float].
        """
        vec = self.lc_embedder.embed_query(text)
        return self._to_1d_list(vec)

    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        """
        Compute embedding vectors for multiple documents.
        Must return a list of 1D list[float] vectors of equal length.
        """
        vecs = self.lc_embedder.embed_documents(docs)
        out = [self._to_1d_list(v) for v in vecs]

        # sanity check: all dims must match
        if out:
            d0 = len(out[0])
            for i, v in enumerate(out):
                if len(v) != d0:
                    raise ValueError(
                        f"Inconsistent embedding dims in batch: vec[0]={d0}, vec[{i}]={len(v)}"
                    )
        return out

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