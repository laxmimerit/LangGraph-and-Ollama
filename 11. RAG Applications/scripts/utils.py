from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from scripts.schemas import ChunkMetadata, RankingKeywords

# =============================================================================
# Configuration
# =============================================================================

# ChromaDB Configuration (from PageRAG - Data Ingestion)
CHROMA_DIR = "chroma_financial_db"
COLLECTION_NAME = "financial_docs"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3"
BASE_URL = "http://localhost:11434"

llm = ChatOllama(
        model=LLM_MODEL,
        base_url=BASE_URL
    )

# Initialize embeddings
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=BASE_URL
)

# Initialize vector store
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)


total_count = vector_store._collection.count()
print(f"[DB] Total documents in database: {total_count}")

def extract_filters(user_query: str):
    """Extract metadata filters from user query."""
    
    llm_structured = llm.with_structured_output(ChunkMetadata)
    
    prompt = f"""Extract metadata filters from the query. Return None for fields not mentioned.

                USER QUERY: {user_query}

                COMPANY MAPPINGS:
                - Amazon/AMZN -> amazon
                - Google/Alphabet/GOOGL/GOOG -> google
                - Apple/AAPL -> apple
                - Microsoft/MSFT -> microsoft
                - Tesla/TSLA -> tesla
                - Nvidia/NVDA -> nvidia
                - Meta/Facebook/FB -> meta

                DOC TYPE:
                - Annual report -> 10-k
                - Quarterly report -> 10-q
                - Current report -> 8-k

                EXAMPLES:
                "Amazon Q3 2024 revenue" -> {{"company_name": "amazon", "doc_type": "10-q", "fiscal_year": 2024, "fiscal_quarter": "q3"}}
                "Apple 2023 annual report" -> {{"company_name": "apple", "doc_type": "10-k", "fiscal_year": 2023}}
                "Tesla profitability" -> {{}}

                Extract metadata:
                """
    
    metadata = llm_structured.invoke(prompt)
    filters = metadata.model_dump(exclude_none=True)
    
    return filters




def generate_ranking_keywords(user_query: str):
    """Generate EXACTLY 5 financial keywords for document ranking."""
    
    prompt = f"""Generate EXACTLY 5 financial keywords from SEC filings terminology.

                USER QUERY: {user_query}

                USE EXACT TERMS FROM 10-K/10-Q FILINGS:

                STATEMENT HEADINGS:
                "consolidated statements of operations", "consolidated balance sheets", "consolidated statements of cash flows", "consolidated statements of stockholders equity"

                INCOME STATEMENT:
                "revenue", "net revenue", "cost of revenue", "gross profit", "operating income", "net income", "earnings per share"

                BALANCE SHEET:
                "total assets", "cash and cash equivalents", "total liabilities", "stockholders equity", "working capital", "long-term debt"

                CASH FLOWS:
                "cash flows from operating activities", "net cash provided by operating activities", "cash flows from investing activities", "free cash flow", "capital expenditures"

                RULES:
                - Return EXACTLY 5 keywords
                - Use exact phrases from SEC filings
                - Match query topic (revenue -> revenue terms, cash -> cash flow terms)
                - Use "cash flows" (plural), "stockholders equity"

                EXAMPLES:
                "revenue analysis" -> ["revenue", "net revenue", "total revenue", "consolidated statements of operations", "net sales"]
                "cash flow performance" -> ["consolidated statements of cash flows", "cash flows from operating activities", "net cash provided by operating activities", "free cash flow", "operating activities"]
                "balance sheet strength" -> ["consolidated balance sheets", "total assets", "stockholders equity", "cash and cash equivalents", "long-term debt"]

                Generate EXACTLY 5 keywords:
                """
    
    llm_structured = llm.with_structured_output(RankingKeywords)
    result = llm_structured.invoke(prompt)
    return result.keywords

def build_search_kwargs(filters, ranking_keywords, k=3):
    """
    Build search kwargs for ChromaDB retriever with proper filter formatting.

    Handles the conversion of flat filter dictionaries to ChromaDB's $and format
    when multiple filters are present, and adds where_document filtering for ranking keywords.

    Args:
        filters: Dictionary of metadata filters
        ranking_keywords: List of keywords to search in document content
        k: Number of results to retrieve

    Returns:
        Dictionary with search_kwargs formatted for ChromaDB

    Examples:
        >>> build_search_kwargs({"company_ticker": "AMZN"}, ["cash flow", "revenue"], k=5)
        {
            "k": 5, 
            "filter": {"company_ticker": "AMZN"},
            "where_document": {"$or": [{"$contains": "cash flow"}, {"$contains": "revenue"}]}
        }

        >>> build_search_kwargs({"company_ticker": "AMZN", "fiscal_year": 2023}, ["liquidity"], k=5)
        {
            "k": 5,
            "filter": {"$and": [{"company_ticker": "AMZN"}, {"fiscal_year": 2023}]},
            "where_document": {"$or": [{"$contains": "liquidity"}]}
        }
    """

    # Build search kwargs
    search_kwargs = {"k": k, 'fetch_k': k * 20}  # fetch_k for reranking

    # Add metadata filters
    if filters:
        # Single filter: use directly
        if len(filters) == 1:
            search_kwargs["filter"] = filters
        # Multiple filters: combine with $and
        else:
            filter_conditions = [{key: value} for key, value in filters.items()]
            search_kwargs["filter"] = {"$and": filter_conditions}

    # Add document content filters using ranking keywords
    if ranking_keywords:
        # Single keyword
        if len(ranking_keywords) == 1:
            search_kwargs["where_document"] = {"$contains": ranking_keywords[0]}
        # Multiple keywords: combine with $or (match ANY keyword)
        else:
            search_kwargs["where_document"] = {
                "$or": [
                    {"$contains": keyword}
                    for keyword in ranking_keywords
                ]
            }

    return search_kwargs

def search_docs(query, filters={}, ranking_keywords=[], k=5):
    """
    Search documents with metadata and content filters.
    
    Args:
        query (str): Search query text
        filters (dict): Metadata filters (e.g., {"company_name": "amazon", "fiscal_year": 2023})
        ranking_keywords (list): Keywords for content filtering (documents must contain at least one)
        k (int): Number of results (default: 5)
    
    Returns:
        list: Matching Document objects
    
    Example:
        docs = search_docs(
            query="Analyze cash flow",
            filters={"company_name": "amazon", "doc_type": "10-k"},
            ranking_keywords=["cash flow", "liquidity"],
            k=10
        )
    """
    search_kwargs = build_search_kwargs(filters, ranking_keywords, k=k)
    
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )
    
    return retriever.invoke(query)

import re
from typing import List
from rank_bm25 import BM25Plus

def extract_headings_with_content(text: str) -> List[str]:
    """
    Extract markdown headings with one paragraph of content after them.
    
    Args:
        text: Document text content
    
    Returns:
        List of extracted heading + content chunks
    """
    chunks = []
    
    # Split by double newlines to get sections
    sections = text.split('\n\n')
    
    i = 0
    while i < len(sections):
        section = sections[i].strip()
        
        # Check if section starts with markdown heading (one or more #)
        if re.match(r'^#+\s+', section):
            # Found a heading
            heading = section
            
            # Get the next paragraph/content after heading
            if i + 1 < len(sections):
                next_content = sections[i + 1].strip()
                chunk = f"{heading}\n\n{next_content}"
                i += 2  # Skip both heading and next content
            else:
                chunk = heading
                i += 1  # Only skip heading
            
            chunks.append(chunk)
        else:
            i += 1
    
    return chunks


def rank_documents_by_keywords(docs, keywords, k=5):
    """
    Rank documents using BM25Plus on heading+content chunks, then return full ranked documents.

    Process:
    1. Extract headings with one paragraph after them from each document
    2. Rank these chunks using BM25Plus with keywords
    3. Return documents sorted by their best chunk score

    Args:
        docs: List of Document objects to rank
        keywords: List of keywords to rank by (e.g., ['consolidated balance sheets', 'total assets'])

    Returns:
        List of Document objects sorted by BM25 score (highest first)

    Example:
        >>> docs = retriever.invoke("Amazon balance sheet")
        >>> ranked = rank_documents_by_keywords(docs, ['consolidated balance sheets', 'total assets'])
    """
    if not docs or not keywords:
        return docs

    # Tokenize keywords (query terms)
    query_tokens = ' '.join(keywords).lower().split()

    # Extract headings+content and track which doc they came from
    doc_chunks = []  # List of (doc_index, chunk_text)
    doc_best_scores = {}  # {doc_index: best_score}
    
    for doc_idx, doc in enumerate(docs):
        # Extract heading+content chunks from document
        chunks = extract_headings_with_content(doc.page_content)
        
        if not chunks:
            # If no headings found, use full content
            chunks = [doc.page_content]
        
        # Store chunks with their document index
        for chunk in chunks:
            doc_chunks.append((doc_idx, chunk))
    
    if not doc_chunks:
        print("[BM25] No chunks extracted")
        return docs
    
    # Tokenize all chunks
    tokenized_chunks = [chunk.lower().split() for _, chunk in doc_chunks]
    
    # Initialize BM25Plus on chunks
    bm25 = BM25Plus(tokenized_chunks)
    
    # Get BM25 scores for all chunks
    chunk_scores = bm25.get_scores(query_tokens)
    
    # Find best score for each document
    for (doc_idx, chunk), score in zip(doc_chunks, chunk_scores):
        if doc_idx not in doc_best_scores or score > doc_best_scores[doc_idx]:
            doc_best_scores[doc_idx] = score
    
    # Rank documents by their best chunk score
    ranked = sorted(
        [(doc_idx, score, docs[doc_idx]) for doc_idx, score in doc_best_scores.items() if score > 0],
        key=lambda x: x[1],
        reverse=True
    )
    
    if not ranked:
        print("[BM25] No documents scored above 0")
        return docs
    
    # Log ranking results
    print(f"[BM25] Ranked {len(ranked)} documents by heading+content chunks")
    for i, (doc_idx, score, _) in enumerate(ranked[:5], 1):
        print(f"  [{i}] Doc {doc_idx}: score={score:.4f}")
    
    # Return sorted documents
    ranked_docs = [doc for _, _, doc in ranked]
    
    # Add unranked docs at the end (docs with score 0)
    unranked_indices = set(range(len(docs))) - {idx for idx, _, _ in ranked}
    ranked_docs.extend([docs[idx] for idx in unranked_indices])
    
    return ranked_docs[:k]


