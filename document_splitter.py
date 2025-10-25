from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

web_loader = WebBaseLoader("https://en.wikipedia.org/wiki/COVID-19")
web_documents = web_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", "", " "])
chunks = text_splitter.split_documents(web_documents)
print("Number of text chunks: ", len(chunks))

for chunk in chunks:
    print(f"Chunk {chunks.index(chunk)} size: {len(chunk.page_content)}")
    print(chunk.page_content)

# there are many more splitters provided by langchain
# like e.g. html, markdown, json etc.