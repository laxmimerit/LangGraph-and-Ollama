"""
Centralized Tools Module
=========================

This module contains all reusable tools for LangGraph agents.
Import and use: import my_tools
"""

from langchain_core.tools import tool
import requests

# =============================================================================
# Weather Tools
# =============================================================================

@tool
def get_weather(location: str) -> str:
    """Get current weather for a location.
    
    Use for queries about weather, temperature, or conditions in any city.
    Examples: "weather in Paris", "temperature in Tokyo", "is it raining in London"
    
    Args:
        location: City name (e.g., "New York", "London", "Tokyo")
        
    Returns:
        Current weather information including temperature and conditions.
    """
    response = requests.get(f"https://wttr.in/{location}?format=j1", timeout=10)
    response.raise_for_status()
    data = response.json()

    return data


# =============================================================================
# Math Tools
# =============================================================================

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.
    
    USE THIS TOOL FOR:
    - Any mathematical calculations or arithmetic operations
    - Queries involving numbers and operators (+, -, *, /, **, %)
    - Questions asking to compute, calculate, or solve math problems
    - Evaluating mathematical expressions
    
    EXAMPLE QUERIES:
    - "What is 2 + 2?"
    - "Calculate 15 times 7"
    - "Solve 100 / 4"
    - "What's 5 to the power of 3?"
    - "Compute 45 * 12 + 30"
    
    DO NOT USE FOR:
    - Word problems without explicit expressions (extract the math first)
    - Questions about mathematical concepts or theory

    Args:
        expression: Math expression like "2 + 2" or "15 * 7" (use standard Python operators)
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        print(f"[TOOL] calculate('{expression}') -> {result}")
        return str(result)
    except Exception as e:
        print(f"[ERROR] calculate failed: {e}")
        return f"Error: {str(e)}"



    return result


@tool
def search_docs(query: str, return_sample: bool = False) -> str:
    """Search technical documentation.
    
    Use for: LangGraph, Ollama, Python, tools, agents, LLMs, technical concepts.
    Examples: "langgraph", "ollama", "python tools", "agent memory"
    
    If not found, call again with return_sample=True to see all available topics.
    
    Args:
        query: Search keywords (e.g., "langgraph", "ollama", "python")
        return_sample: If True, returns all available topics (default: False)
    """
    docs = {
        "langgraph": "LangGraph: Framework for stateful agents with LLMs",
        "ollama": "Ollama: Run LLMs locally on your machine",
        "python": "Python: High-level programming language",
        "tools": "Tools let LLMs call functions to get information",
        "memory": "Memory allows agents to remember conversation context",
    }
    
    if return_sample:
        sample_output = "Available documentation topics:\n" + "\n".join(
            [f"- {key}: {value}" for key, value in docs.items()]
        )
        print(f"[TOOL] search_docs(return_sample=True) -> Returning all {len(docs)} topics")
        return sample_output
    
    result = docs.get(query.lower(), "Not found")
    print(f"[TOOL] search_docs('{query}', return_sample=False) -> {result}")
    
    if result == "Not found":
        result += "\nCall again with return_sample=True to see all topics."
    
    return result