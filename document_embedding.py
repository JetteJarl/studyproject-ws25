from typing import Iterable, List
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings_model(model_name: str) -> HuggingFaceEmbeddings:
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings_model

def embed_texts(embeddings_model: HuggingFaceEmbeddings, texts: Iterable[str]) -> List[List[float]]:
    embeddings = embeddings_model.embed_documents(list(texts))
    return embeddings

# there are many more embeddings provided by huggingface, openai, google, etc.
# check out the leaderboard of embedding models