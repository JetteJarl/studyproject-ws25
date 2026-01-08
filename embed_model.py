from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings_model(model: str) -> tuple[HuggingFaceEmbeddings, str]:
    """
    Create an embedding model instance.

    Args:
        model: Name of the sentence-transformers model.

    Returns:
        A LangChain Embeddings implementation and the model name.
    """
    # HuggingFaceEmbeddings loads a local/remote model by name; caching is handled internally.
    embeddings_model = HuggingFaceEmbeddings(model=model)
    return embeddings_model, model

# there are many more embeddings provided by huggingface, openai, google, etc.
# check out the leaderboard of embedding models