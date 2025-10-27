from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings_model(model_name: str) -> HuggingFaceEmbeddings:
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings_model

# there are many more embeddings provided by huggingface, openai, google, etc.
# check out the leaderboard of embedding models