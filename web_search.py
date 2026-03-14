
# web_search.py
# Searches the internet using Tavily API
# Returns LIVE results (not stored in FAISS)

import os
from typing import List
from models import WebSearchResult


MAX_WEB_RESULTS = 3       # how many results to fetch
MAX_CONTENT_LENGTH = 500  # max characters per result


def search_web(query: str) -> List[WebSearchResult]:
    # Searches web using Tavily and returns results
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found! Get free key at tavily.com")

    print(f"Searching web for: '{query}'")

    from langchain_community.tools.tavily_search import TavilySearchResults

    search_tool = TavilySearchResults(max_results=MAX_WEB_RESULTS, api_key=api_key)
    raw_results = search_tool.invoke(query)

    web_results = []
    for i, result in enumerate(raw_results):
        content = result.get('content', '')
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "..."

        web_results.append(WebSearchResult(
            title=result.get('title', f'Web Result {i+1}'),
            content=content,
            url=result.get('url', ''),
            score=result.get('score', 0.0)
        ))

    print(f"Found {len(web_results)} web results")
    return web_results


def format_web_results(results: List[WebSearchResult]) -> str:
    # Formats web results into a string for the LLM prompt
    if not results:
        return "No web results available."

    formatted = []
    for i, result in enumerate(results):
        formatted.append(
            f"[Web Source {i+1}]\n"
            f"Title: {result.title}\n"
            f"Content: {result.content}\n"
            f"URL: {result.url}"
        )
    return "\n\n".join(formatted)
