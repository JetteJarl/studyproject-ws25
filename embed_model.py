from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings_model(model_name: str) -> HuggingFaceEmbeddings:
    """
    Create an embedding model instance.

    Args:
        model_name: Name of the sentence-transformers model.

    Returns:
        A LangChain Embeddings implementation.
    """
    # HuggingFaceEmbeddings loads a local/remote model by name; caching is handled internally.
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings_model

# there are many more embeddings provided by huggingface, openai, google, etc.
# check out the leaderboard of embedding models