import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any

from app.tools.web_search import duckduckgo_search
from app.tools.fetcher import fetch_url, fetch_pdf
from app.agent.state import AgentState
from app.agent.planner import create_plan


mcp = FastMCP(
    name="autonomous-research-agent",
    instructions=(
        "You are an autonomous research agent. "
        "You plan, use tools, store memory, and produce "
        "structured, citation-backed research reports."
    ),
)

@mcp.tool()
async def web_search(query: str) -> list[str]:
    """
    Search the web for relevant sources.
    Use this when the topic requires external information.
    """
    return duckduckgo_search(query)

@mcp.tool()
async def fetch_page_content(url: str) -> str:
    """
    Fetch text content from a URL (HTML, Blog, etc.).
    """
    return fetch_url(url)

@mcp.tool()
async def fetch_pdf_content(url: str) -> str:
    """
    Fetch text content from a PDF URL.
    """
    return fetch_pdf(url)
@mcp.tool()
async def query_rag(query: str, k: int = 3) -> list:
    """
    Query the internal RAG store for relevant documents.
    """
    return rag.retrieve(query, k=k)

@mcp.resource("rag://knowledge")
def get_rag_knowledge() -> str:
    """
    Get all text chunks currently in the RAG store.
    """
    return "\n---\n".join(rag.text_chunks)

from app.rag import RAGStore
from app.agent.reasoning import synthesize_report
import asyncio
import sys

# Global RAG store (simulating long-term memory)
rag = RAGStore()

@mcp.tool()
async def run_research_task(query: str) -> dict:
    """
    Run the autonomous research agent on a given query.
    This orchestrates the planning, searching, fetching, indexing, and reporting.
    """
    state = AgentState(query=query)

    # 1. Plan
    try:
        state.plan = create_plan(query)
        # print(f"DEBUG: Plan created: {state.plan}", file=sys.stderr)
    except Exception as e:
        # print(f"DEBUG: Plan creation failed: {e}", file=sys.stderr)
        return {"error": f"Plan creation failed: {e}"}

    if not state.plan:
        return {"error": "Failed to generate a plan"}
        
    state.observations.append(f"Plan created with {len(state.plan)} steps.")

    # 2. Execute steps (Limit to first 3 steps for resume-demo speed)
    for i, step in enumerate(state.plan[:3]): 
        # print(f"DEBUG: Executing step {i}: {step}", file=sys.stderr)
        # A. Search
        try:
            search_results = await web_search(step)
            # print(f"DEBUG: Search results for '{step}': {len(search_results)}", file=sys.stderr)
            state.observations.append(f"Searched: {step}, found {len(search_results)} URLs.")
        except Exception as e:
            # print(f"DEBUG: Search failed: {e}", file=sys.stderr)
            state.observations.append(f"Search failed for {step}: {e}")
            continue
        
        # B. Fetch & Index
        for url in search_results[:2]: # Limit to top 2 URLs per step
            try:
                content = ""
                if url.endswith(".pdf"):
                    content = await fetch_pdf_content(url)
                else:
                    content = await fetch_page_content(url)
                
                if content:
                    rag.add_document(content, source=url)
                    state.sources.append({"url": url, "status": "indexed"})
            except Exception as e:
                state.observations.append(f"Failed to fetch {url}: {e}")

    # 3. Retrieve relevant info
    retrieved_docs = rag.retrieve(query, k=3)
    
    # 4. Synthesize Report
    if not retrieved_docs:
         report = "No relevant information found to synthesize a report."
    else:
         report = synthesize_report(query, retrieved_docs)

    return {
        "query": state.query,
        "plan": state.plan,
        "sources": state.sources,
        "observations": state.observations,
        "report": report
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run directly as a CLI entrypoint if a query is provided
        query_arg = sys.argv[1]
        print(f"Running research task for: {query_arg}", file=sys.stderr)
        import asyncio
        result = asyncio.run(run_research_task(query_arg))
        import json
        print(json.dumps(result, indent=2))
    else:
        # Default: Initialize and run the MCP server
        mcp.run()
