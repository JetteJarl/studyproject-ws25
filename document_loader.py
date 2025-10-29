from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

def load_web_page(url: str) -> List[Document]:
    """
    Fetch a web page and wrap its extracted text as a LangChain Document list.

    Args:
        url: Fully qualified URL to fetch.

    Returns:
        A list with a single Document containing page text and source metadata.

    Raises:
        requests.HTTPError: If the HTTP request fails with a bad status.
    """
    # Download the page contents
    loader = WebBaseLoader(url, raise_for_status=True)
    document = loader.load()
    return document

# there are many more loaders provided by langchain
# like e.g. txt, pdf, csv, etc.