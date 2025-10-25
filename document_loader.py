from langchain_community.document_loaders import TextLoader

loader = TextLoader("climate_change.txt")
documents = loader.load()
print(documents)
print(documents[0].page_content)