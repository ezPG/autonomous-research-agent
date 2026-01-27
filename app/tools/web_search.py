from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing import List

def duckduckgo_search(query: str, max_results: int = 5) -> List[str]:
    """
    Search the web using DuckDuckGo via LangChain.
    Returns a list of result snippets/URLs.
    """
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
    # results returns a list of dicts with 'snippet', 'title', 'link'
    results = wrapper.results(query, max_results=max_results)
    return [r['link'] for r in results if 'link' in r]
