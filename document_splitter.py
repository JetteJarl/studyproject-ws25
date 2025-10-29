from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[list[str]] = None,
) -> List[Document]:
    """
    Split a list of documents into smaller chunks suitable for vector indexing.

    Args:
        docs: Source documents to split.
        chunk_size: Target character length of each chunk (approximate).
        chunk_overlap: Overlap in characters between adjacent chunks to preserve context.
        separators: Preferred boundaries used by the splitter when breaking text.
            Defaults to the paragraph/newline/space fallback sequence.

    Returns:
        A list of chunked Document objects.
    """
    if separators is None:
        # Default separator order: prefer paragraphs, then lines, then spaces, then hard wrap
        separators = ["\n\n", "\n", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    chunks = splitter.split_documents(docs)
    return chunks

# there are many more splitters provided by langchain
# like e.g. html, markdown, json etc.