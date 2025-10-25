from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader

txt_loader = TextLoader("climate_change.txt")
txt_documents = txt_loader.load()
print(txt_documents)
print(txt_documents[0].page_content)

web_loader = WebBaseLoader("https://en.wikipedia.org/wiki/COVID-19")
web_documents = web_loader.load()
print(web_documents)
print(web_documents[0].page_content)

# there are many more loaders provided by langchain
# like e.g. pdf, csv, etc.