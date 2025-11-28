from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import BaseRagasEmbeddings

class LocalOllamaRagasLLM(LangchainLLMWrapper):
    """
    LangChain-to-Ragas adapter for a local Ollama chat model.

    This wrapper exposes a LangChain OllamaLLM instance through the Ragas LLM
    interface so it can be used as a judge/model within Ragas metrics.

    Parameters:
        model: Name of the Ollama model available locally (e.g., "llama3", "mistral").
        base_url: HTTP endpoint of the Ollama server (e.g., "http://localhost:11434").

    Attributes:
        langchain_llm: Underlying LangChain OllamaLLM client.
        bypass_temperature: Hint for Ragas to ignore temperature where applicable.
        is_finished_parser: Optional callable to detect end-of-generation (unused here).
        run_config: Optional per-run configuration set via set_run_config.
    """
    def __init__(self, model: str, base_url: str):
        self.langchain_llm = OllamaLLM(model=model, base_url=base_url)
        self.bypass_temperature = True
        self.is_finished_parser = None

    async def _agenerate(self, prompt: str) -> str:
        """
        Asynchronously generate a completion for a given prompt.

        Parameters:
            prompt: Input text prompt to be sent to the Ollama model.

        Returns:
            The generated text string.
        """
        return await self.langchain_llm.apredict(prompt)

    def set_run_config(self, config):
        """
        Attach a per-run configuration object for downstream consumers.

        Parameters:
            config: Arbitrary configuration object stored on the instance.

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