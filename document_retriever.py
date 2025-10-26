from typing import List
from langchain_core.documents import Document

def retrieve_docs(retriever, query: str) -> List[Document]:
    chunks = retriever.invoke(query)
    return chunks