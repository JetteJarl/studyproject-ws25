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
        """
        Create a Mistral-backed async LLM adapter for Ragas.

        This class is used internally by :class:`MistralRagasLLM` to satisfy the
        interface that :class:`ragas.llms.base.LangchainLLMWrapper` expects
        (namely :meth:`agenerate_prompt`).

        Args:
            model: Mistral model name (e.g., ``"open-mixtral-8x7b"``).
            max_concurrency: Maximum number of in-flight Mistral requests.
                Keeping this at 1 reduces rate limiting and tail-latency timeouts.
            request_timeout_s: Hard timeout per request in seconds. If the request
                exceeds this, it is treated as a failure and can be retried.
            max_retries: Maximum number of retries for rate limits/timeouts before
                the exception is propagated.
            base_backoff_s: Base delay for exponential backoff between retries.
            max_backoff_s: Maximum delay (cap) for exponential backoff.
        """
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
        """
        Convert a prompt-like object into a plain string.

        Ragas/LangChain may pass raw strings or PromptValue-like objects that
        implement ``to_string()``.

        Args:
            p: Prompt input (string or PromptValue-like object).

        Returns:
            Prompt text as a string.
        """
        if isinstance(p, str):
            return p
        to_string = getattr(p, "to_string", None)
        if callable(to_string):
            return to_string()
        return str(p)

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        """
        Best-effort detection of rate-limit errors from the Mistral SDK.

        The Mistral SDK may raise different exception types depending on version.
        This helper checks the exception message for common indicators.

        Args:
            exc: The exception raised during a request.

        Returns:
            True if the exception appears to be a rate-limit (HTTP 429), else False.
        """
        msg = str(exc)
        return "Status 429" in msg or "rate limit" in msg.lower() or "rate_limited" in msg.lower()

    def _complete_one_sync(self, prompt_text: str) -> str:
        """
        Execute one synchronous chat completion against the Mistral API.

        A strict system message and ``temperature=0`` are used to make the judge
        model more deterministic and more likely to return schema-compliant JSON
        when Ragas requests structured outputs.

        Args:
            prompt_text: Fully rendered prompt text.

        Returns:
            The model's completion text.
        """
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
        """
        Execute one completion with concurrency limiting, retries, and a hard timeout.

        This method:
          - limits concurrency via a semaphore
          - applies a hard timeout around the API call
          - retries on timeouts and rate limits using exponential backoff with jitter

        Args:
            prompt_text: Fully rendered prompt text.

        Returns:
            The model's completion text.

        Raises:
            TimeoutError: If the request times out more than ``max_retries`` times.
            Exception: Propagates non-rate-limit errors immediately, or after exhausting
                retries for rate-limit errors.
        """
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
        """
        Generate completions for a batch of prompts (async).

        This is the primary entry point used by Ragas via ``LangchainLLMWrapper``.
        It returns a LangChain ``LLMResult`` with one ``Generation`` per prompt.

        Args:
            prompts: A list of prompt objects (strings or PromptValue-like).
            **kwargs: Extra generation parameters (ignored; present for compatibility).

        Returns:
            A LangChain ``LLMResult`` where ``generations[i][0].text`` contains the
            completion for ``prompts[i]``.
        """
        prompt_texts = [self._prompt_to_text(p) for p in prompts]
        outputs = await asyncio.gather(*[self._complete_one(t) for t in prompt_texts])
        generations = [[Generation(text=o)] for o in outputs]
        return LLMResult(generations=generations)

    async def apredict(self, prompt: str) -> str:
        """
        Convenience method to generate a single completion (async).

        Some LangChain/Ragas code paths call ``apredict`` instead of
        ``agenerate_prompt``. This delegates to :meth:`_complete_one`.

        Args:
            prompt: Prompt text.

        Returns:
            The completion text.
        """
        return await self._complete_one(prompt)


class MistralRagasLLM(LangchainLLMWrapper):
    """
    Ragas LLM wrapper using the Mistral API directly (no Ollama).

    Parameters:
        model: Mistral model name, e.g. "open-mixtral-8x7b"
    """
    def __init__(self, model: str):
        """
        Create a Ragas-compatible judge LLM backed by Mistral.

        Args:
            model: Mistral model name (e.g., ``"open-mixtral-8x7b"``).

        Notes:
            Ragas expects a ``LangchainLLMWrapper`` instance with a ``langchain_llm``
            attribute that implements the relevant async generation methods.
            This wrapper provides that via :class:`_MistralAsyncAdapter`.
        """
        # max_concurrency=1 is slow but stable; raise to 2 only if you stop seeing 429/TimeoutError.
        self.langchain_llm = _MistralAsyncAdapter(model=model)
        self.bypass_temperature = True
        self.is_finished_parser = None
        self.run_config = None

    def set_run_config(self, config: Any) -> None:
        """
        Attach a per-run configuration object (as expected by Ragas).

        Ragas may call this to pass runtime configuration (timeouts, logging additionally,
        tracing info, etc.) to the LLM wrapper. The object is stored and not interpreted
        further by this class.

        Args:
            config: Arbitrary configuration object provided by Ragas.

        Returns:
            None.
        """
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
        """
        Create an embeddings backend for Ragas using a HuggingFace sentence-transformer.

        Args:
            model_name: Hugging Face embeddings model identifier.
        """
        self.lc_embedder = HuggingFaceEmbeddings(model_name=model_name)

    def _to_1d_list(self, vec: Any) -> list[float]:
        """
        Normalize an embedding vector into a 1D Python list of floats.

        This helps avoid shape-related errors (e.g., (1, d) vs (d,)) and ensures
        consistent output types for NumPy-based similarity computations.

        Args:
            vec: Embedding vector returned by the underlying embeddings model.

        Returns:
            A 1D list of floats.

        Raises:
            ValueError: If the input cannot be coerced into a 1D vector.
        """
        arr = np.asarray(vec, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D embedding vector, got shape {arr.shape}")
        return arr.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Compute an embedding vector for a single query string.

        Args:
            text: Input text to embed.

        Returns:
            A 1D embedding vector as ``list[float]``.
        """
        vec = self.lc_embedder.embed_query(text)
        return self._to_1d_list(vec)

    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        """
        Compute embedding vectors for multiple documents.

        Args:
            docs: List of document strings.

        Returns:
            A list of 1D embedding vectors (``list[list[float]]``), one per document.

        Raises:
            ValueError: If the underlying embedder returns inconsistent vector dimensions.
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

        Args:
            query: Input query text.

        Returns:
            A 1D embedding vector as ``list[float]``.

        Notes:
            The underlying LangChain embedder is synchronous; this method currently calls it
            directly. If you need true non-blocking behavior, wrap the call in
            ``asyncio.to_thread(...)``.
        """
        return self.lc_embedder.embed_query(query)

    async def aembed_documents(self, docs: list[str]):
        """
        Asynchronously compute embeddings for multiple documents.

        Args:
            docs: List of document strings.

        Returns:
            A list of embedding vectors (``list[list[float]]``), one per document.

        Notes:
            The underlying LangChain embedder is synchronous; this method currently calls it
            directly. If you need true non-blocking behavior, wrap the call in
            ``asyncio.to_thread(...)``.
        """
        return self.lc_embedder.embed_documents(docs)

    async def embed_text(self, text: str):
        """
        Asynchronously compute an embedding for a single text.

        This is effectively an alias for :meth:`embed_query`, provided for compatibility
        with Ragas versions that call ``embed_text``.

        Args:
            text: Input text.

        Returns:
            A 1D embedding vector as ``list[float]``.

        Notes:
            The underlying LangChain embedder is synchronous; this method currently calls it
            directly. If you need true non-blocking behavior, wrap the call in
            ``asyncio.to_thread(...)``.
        """
        return self.lc_embedder.embed_query(text)