from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

def load_web_page(url: str) -> List[Document]:
    loader = WebBaseLoader(url)
    document = loader.load()
    return document

# there are many more loaders provided by langchain
# like e.g. txt, pdf, csv, etc.