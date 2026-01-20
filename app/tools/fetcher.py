from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
import tempfile
import requests
import os

# Set User-Agent for LangChain loaders
os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def fetch_url(url: str) -> str:
    """
    Fetch text content from a URL using LangChain's WebBaseLoader.
    """
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error fetching URL: {e}"

def fetch_pdf(url: str) -> str:
    """
    Fetch text content from a PDF URL using LangChain's PyMuPDFLoader.
    """
    try:
        # Load PDF into a temporary file if it's a URL
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        try:
            loader = PyMuPDFLoader(tmp_path)
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        return f"Error fetching PDF: {e}"
