from typing import List
from langchain_core.documents import Document

def retrieve_docs(retriever, query: str) -> List[Document]:
    """
    Retrieve relevant documents for a query from a retriever.

    Args:
        retriever: Vector-based retriever.
        query: The user question or search query.

    Returns:
        A list of relevant Documents.
    """
    chunks = retriever.invoke(query)
    return chunks