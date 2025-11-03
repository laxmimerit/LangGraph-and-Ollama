"""
LangChain tools for RAG applications.
Tool implementation functions that can be wrapped with @tool decorator in notebooks.
"""

import os
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from scripts import utils

# =============================================================================
# Configuration - Set these before using the tools
# =============================================================================
LLM_MODEL = "qwen3"
BASE_URL = "http://localhost:11434"

# Initialize LLM
llm = ChatOllama(
    model=LLM_MODEL,
    base_url=BASE_URL
)


@tool
def retrieve_docs(query: str, k=5) -> str:
    """
    Retrieve relevant financial documents from ChromaDB.
    Extracts filters from query and retrieves matching documents.

    Args:
        query: The search query (e.g., "What was Amazon's revenue in Q2 2025?")
        k: Number of documents to retrieve

    Returns:
        Retrieved documents with metadata as formatted string
    """
    print(f"\n[TOOL] retrieve_docs called")
    print(f"[QUERY] {query}")

    # Extract filters from query
    filters = utils.extract_filters(query)
    ranking_keywords = utils.generate_ranking_keywords(query)

    results = utils.search_docs(query, filters, ranking_keywords, k=20)

    docs = utils.rank_documents_by_keywords(results, ranking_keywords, k=k)

    print(f"[RETRIEVED] {len(docs)} documents")

    # Handle empty results
    if len(docs) == 0:
        return "No documents found. Try rephrasing your query or using different filters."

    # Format results
    retrieved_text = []

    for i, doc in enumerate(docs, 1):
        doc_text = [f"\n--- Document {i} ---"]

        # Add all metadata
        for key, value in doc.metadata.items():
            doc_text.append(f"{key}: {value}")

        # Add content
        doc_text.append(f"\nContent:\n{doc.page_content}")

        retrieved_text.append("\n".join(doc_text))

    retrieved_text = "\n".join(retrieved_text)

    # store retrieved text for debugging
    os.makedirs("debug_logs", exist_ok=True)
    with open("debug_logs/retrieved_reranked_docs.md", "w", encoding="utf-8") as f:
        f.write(retrieved_text)

    return retrieved_text


# DuckDuckGo search integration
from ddgs import DDGS

@tool
def web_search(query: str, num_results: int = 10) -> str:
    """Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 10)
    
    Returns:
        Formatted search results with titles, descriptions, and URLs
    """
    
    try:
        results = list(DDGS().text(query=query,
                                   max_results=num_results,
                                   region="us-en",
                                   timelimit="d",
                                   backend="google, bing, brave, yahoo, wikipedia, duckduckgo"))
        
        if not results:
            return f"No results found for '{query}'"
        
        formatted_results = [f"Search Results for '{query}':\n"]
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description available')
            href = result.get('href', '')
            formatted_results.append(f"{i}. **{title}**\n   {body}\n   {href}")
        
        return "\n\n".join(formatted_results)
    
    except Exception as e:
        return f"Search error: {str(e)}"