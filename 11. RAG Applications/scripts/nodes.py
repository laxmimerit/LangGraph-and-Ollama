"""
Self-RAG Nodes
==============

Reusable nodes for Self-RAG flow that can be used across different RAG implementations.
"""

from typing import Dict
import os

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from scripts import my_tools

# =============================================================================
# Configuration
# =============================================================================

LLM_MODEL = "qwen3"
BASE_URL = "http://localhost:11434"

llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL)

# =============================================================================
# Pydantic Schemas for Structured Outputs
# =============================================================================

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the query, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses query."""
    binary_score: str = Field(
        description="Answer addresses the query, 'yes' or 'no'"
    )

class SearchQueries(BaseModel):
    """Search queries for retrieving missing information."""
    search_queries: list[str] = Field(
        description="1-3 search queries to retrieve missing information"
    )

# =============================================================================
# LangGraph Nodes
# =============================================================================

def retrieve_node(state) -> Dict:
    """Retrieve documents based on user query."""
    print("\n[NODE] Retrieve - Fetching documents")

    query = state["query"]
    rewritten_queries = state.get("rewritten_queries", [])

    # Use rewritten queries if present, otherwise use original query
    queries_to_search = rewritten_queries if rewritten_queries else [query]

    all_results = []
    for idx, search_query in enumerate(queries_to_search, 1):
        print(f"[QUERY {idx}] {search_query}")
        result = my_tools.retrieve_docs.invoke({'query': search_query, 'k': 3})
        all_results.append(f"## Query {idx}: {search_query}\n\n### Retrieved Documents:\n{result}")

    combined_result = "\n\n" + "\n\n".join(all_results)
    print(f"[RETRIEVED] Documents fetched for {len(queries_to_search)} queries")

    # Save for debugging
    os.makedirs("debug_logs", exist_ok=True)
    with open("debug_logs/self_rag_retrieved_docs.md", "w", encoding="utf-8") as f:
        f.write(f"Original Query: {query}\n")
        if rewritten_queries:
            f.write(f"Rewritten Queries: {rewritten_queries}\n\n")
        f.write(combined_result)

    return {
        "documents": combined_result,
        "query": query
    }

def grade_documents_node(state) -> Dict:
    """Grade document relevance and filter out irrelevant ones."""
    print("\n[NODE] Grade Documents - Evaluating document relevance")

    query = state["query"]
    documents = state.get("documents", "")

    # Create structured output LLM
    llm_structured = llm.with_structured_output(GradeDocuments)

    system_prompt = """You are a grader assessing relevance of retrieved documents to a user query.

It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

If the document contains keyword(s) or semantic meaning related to the user query, grade it as relevant.

Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the query."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Retrieved documents:\n\n{documents}\n\nUser query: {query}")
    ]

    response = llm_structured.invoke(messages)

    print(f"[GRADE] Relevance: {response.binary_score}")

    if response.binary_score == "yes":
        return {
            "filtered_documents": documents,
            "query": query
        }
    else:
        return {
            "filtered_documents": "",
            "query": query
        }

def generate_node(state) -> Dict:
    """Generate answer based on retrieved documents."""
    print("\n[NODE] Generate - Creating answer")

    query = state["query"]
    documents = state.get("filtered_documents", "")

    system_prompt = """You are a financial document analyst providing detailed, accurate answers.

                        **OUTPUT FORMAT:**
                        Write a comprehensive answer (200-300 words) in **MARKDOWN** format:
                        - Use ## headings for sections
                        - Use **bold** for emphasis
                        - Use bullet points or numbered lists
                        - Include inline citations like [1], [2] where applicable

                        **GUIDELINES:**
                        - Base your answer ONLY on the provided documents
                        - Be specific with numbers, dates, and metrics
                        - If information is missing, acknowledge it
                        - Use proper financial terminology

                        **CITATIONS:**
                        At the end, list references in this format:
                        **References:**
                        1. Company: x, Year: y, Quarter: z, Page: n
                        """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Documents:\n\n{documents}\n\nQuery: {query}")
    ]

    response = llm.invoke(messages)
    generation = response.content

    print(f"[GENERATED] Answer created ({len(generation)} chars)")

    # Save for debugging
    with open("debug_logs/self_rag_generation.md", "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n\n")
        f.write(generation)

    return {
        "generation": generation,
        "documents": documents,
        "query": query
    }

def transform_query_node(state) -> Dict:
    """Transform the query to produce a better query."""
    print("\n[NODE] Transform Query - Rewriting query")

    query = state["query"]
    rewritten_queries = state.get("rewritten_queries", [])

    # Create structured output LLM
    llm_structured = llm.with_structured_output(SearchQueries)

    system_prompt = """You are a query re-writer that decomposes complex queries into focused search queries optimized for vectorstore retrieval.

**DECOMPOSITION STRATEGY:**
Break down the original query into 1-3 specific, focused queries where each query targets:
- A single company (e.g., "Amazon revenue 2023" vs "Google revenue 2023")
- A specific time period (e.g., "Q1 2024" vs "Q2 2024")
- A specific metric or aspect (e.g., "revenue" vs "net income")
- A specific document section (e.g., "risk factors" vs "business overview")

**GUIDELINES:**
- Expand abbreviations (e.g., "rev" -> "revenue", "GOOGL" -> "Google")
- Add financial context if missing
- Make each query self-contained and specific
- Keep queries concise but clear (5-10 words each)
- Avoid repeating previously tried queries

**EXAMPLES:**
- "Compare Apple and Google revenue in 2024 Q1" -> ["Apple total revenue Q1 2024", "Google total revenue Q1 2024"]
- "Amazon's revenue growth from 2022 to 2024" -> ["Amazon revenue 2022", "Amazon revenue 2023", "Amazon revenue 2024"]
- "What were the main risks for Microsoft in 2023?" -> ["Microsoft risk factors 2023", "Microsoft business challenges 2023"]"""

    query_context = f"Original query: {query}"
    if rewritten_queries:
        query_context += f"\n\nPreviously tried queries:\n" + "\n".join(f"- {q}" for q in rewritten_queries)
    query_context += "\n\nGenerate 1-3 focused search queries that decompose the original query. Each query should target a specific aspect."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query_context)
    ]

    response = llm_structured.invoke(messages)
    new_queries = response.search_queries

    print(f"[ORIGINAL] {query}")
    print(f"[DECOMPOSED QUERIES] {new_queries}")

    return {
        "query": query,
        "rewritten_queries": new_queries
    }

# =============================================================================
# Router Logic
# =============================================================================

def decide_to_generate(state) -> str:
    """Decide whether to generate answer or transform query."""
    print("\n[ROUTER] Assess graded documents")

    filtered_documents = state.get("filtered_documents", "")

    if not filtered_documents or filtered_documents.strip() == "":
        print("[DECISION] No relevant documents - transforming query")
        return "transform_query"
    else:
        print("[DECISION] Have relevant documents - generating answer")
        return "generate"

def grade_generation_v_documents_and_query(state) -> str:
    """Check for hallucinations and whether answer addresses query."""
    print("\n[ROUTER] Check hallucinations and answer quality")

    query = state["query"]
    documents = state.get("filtered_documents", "")
    generation = state.get("generation", "")

    # Step 1: Check hallucinations
    print("[CHECK] Hallucinations")
    llm_hallucination = llm.with_structured_output(GradeHallucinations)

    hallucination_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.

Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

    messages = [
        SystemMessage(content=hallucination_prompt),
        HumanMessage(content=f"Set of facts:\n\n{documents}\n\nLLM generation: {generation}")
    ]

    hallucination_response = llm_hallucination.invoke(messages)
    hallucination_grade = hallucination_response.binary_score

    if hallucination_grade == "yes":
        print("[DECISION] Generation is grounded in documents")

        # Step 2: Check if answer addresses query
        print("[CHECK] Answer quality")
        llm_answer = llm.with_structured_output(GradeAnswer)

        answer_prompt = """You are a grader assessing whether an answer addresses / resolves a query.

Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the query."""

        messages = [
            SystemMessage(content=answer_prompt),
            HumanMessage(content=f"User query:\n\n{query}\n\nLLM generation: {generation}")
        ]

        answer_response = llm_answer.invoke(messages)
        answer_grade = answer_response.binary_score

        if answer_grade == "yes":
            print("[DECISION] Generation addresses query - USEFUL")
            return "useful"
        else:
            print("[DECISION] Generation does NOT address query - NOT USEFUL")
            return "not useful"
    else:
        print("[DECISION] Generation NOT grounded in documents - NOT SUPPORTED (regenerate)")
        return "not supported"
