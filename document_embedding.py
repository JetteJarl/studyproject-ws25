from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

web_loader = WebBaseLoader("https://en.wikipedia.org/wiki/COVID-19")
web_documents = web_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", "", " "])
chunks = text_splitter.split_documents(web_documents)
print("Number of text chunks: ", len(chunks))

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = embeddings_model.embed_documents([chunk.page_content for chunk in chunks])
print("Embedding vector dimension size: ", len(embeddings[0]))

# there are many more embeddings provided by huggingface, openai, google, etc.
# check out the leaderboard of embedding models