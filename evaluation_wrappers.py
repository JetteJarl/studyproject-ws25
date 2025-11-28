from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import BaseRagasEmbeddings

class LocalOllamaRagasLLM(LangchainLLMWrapper):
    def __init__(self, model: str, base_url: str):
        self.langchain_llm = OllamaLLM(model=model, base_url=base_url)
        self.bypass_temperature = True
        self.is_finished_parser = None

    async def _agenerate(self, prompt: str) -> str:
        return await self.langchain_llm.apredict(prompt)

    def set_run_config(self, config):
        self.run_config = config

class RagasHuggingFaceWrapper(BaseRagasEmbeddings):
    """
    Adapter for ragas 0.3.9 embedding interface.
    Satisfies ALL required methods:
        - embed_query         (sync)
        - embed_documents     (sync)
        - aembed_query        (async)
        - aembed_documents    (async)
        - embed_text          (async)
    """

    def __init__(self, model_name: str):
        self.lc_embedder = HuggingFaceEmbeddings(model_name=model_name)

    def embed_query(self, query: str):
        return self.lc_embedder.embed_query(query)

    def embed_documents(self, docs: list[str]):
        return self.lc_embedder.embed_documents(docs)

    async def aembed_query(self, query: str):
        return self.lc_embedder.embed_query(query)

    async def aembed_documents(self, docs: list[str]):
        return self.lc_embedder.embed_documents(docs)

    async def embed_text(self, text: str):
        return self.lc_embedder.embed_query(text)