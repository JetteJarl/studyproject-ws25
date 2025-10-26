from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[list[str]] = None,
) -> List[Document]:
    if separators is None:
        separators = ["\n\n", "\n", "", " "]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
    )
    chunks = splitter.split_documents(docs)
    return chunks

# there are many more splitters provided by langchain
# like e.g. html, markdown, json etc.