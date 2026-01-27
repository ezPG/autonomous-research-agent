import os
import sys
import json
import asyncio
from typing import List, Dict, Any

from mcp.server.fastmcp import FastMCP
from app.rag import RAGStore

from app.tools.web_search import duckduckgo_search
from app.tools.fetcher import fetch_url, fetch_pdf
# from app.agent.state import AgentState
# from app.agent.planner import classify_intent
# from app.agent.reasoning import synthesize_report, generate_chat_response


mcp = FastMCP(
    name="autonomous-research-agent",
    instructions=(
        "You are an autonomous research agent."
        "You plan, use tools, store memory, and produce structured, citation-backed research reports."
    ),
)

# Global RAG store
rag = RAGStore()

@mcp.resource("rag://knowledge")
def get_rag_knowledge() -> str:
    """
    Get all text chunks currently in the RAG store.
    """
    return "\n---\n".join(rag.text_chunks)

@mcp.tool()
async def web_search(query: str) -> str:
    """
    Search the web for relevant sources.
    Returns a list of URLs as a JSON string.
    """
    results = duckduckgo_search(query)
    return json.dumps(results)

@mcp.tool()
async def fetch_page_content(url: str) -> str:
    """
    Fetch text content from a URL (HTML, Blog, etc.) and index it.
    """
    content = fetch_url(url)
    if content:
        rag.add_document(content, source=url)
        return f"Fetched and indexed content from {url} (Length: {len(content)})"
    return f"Failed to fetch content from {url}"

@mcp.tool()
async def fetch_pdf_content(url: str) -> str:
    """
    Fetch text content from a PDF URL and index it.
    """
    content = fetch_pdf(url)
    if content:
        rag.add_document(content, source=url)
        return f"Fetched and indexed PDF content from {url} (Length: {len(content)})"
    return f"Failed to fetch PDF content from {url}"

@mcp.tool()
async def query_rag(query: str, k: int = 5) -> str:
    """
    Query the internal RAG store for relevant documents.
    Returns results as a JSON string.
    """
    results = rag.retrieve(query, k=k)
    return json.dumps(results)

@mcp.tool()
async def index_text(text: str, source: str = "manual") -> str:
    """
    Directly index a piece of text into the RAG store.
    """
    rag.add_document(text, source=source)
    return f"Indexed text from {source} (Length: {len(text)})"

@mcp.tool()
async def clear_rag() -> str:
    """
    Clear all documents and index from the RAG store.
    """
    rag.clear()
    return "RAG store cleared successfully."


if __name__ == "__main__":
        mcp.run()
