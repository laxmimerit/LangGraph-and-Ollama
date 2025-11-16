# # RAG Data Retrieval and Re-Ranking

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from scripts.schemas import ChunkMetadata, RankingKeywords
import re
from rank_bm25 import BM25Plus

# =============================================================================
# Configuration
# =============================================================================

# ChromaDB Configuration (from PageRAG - Data Ingestion)
CHROMA_DIR = "chroma_financial_db"
COLLECTION_NAME = "financial_docs"
EMBEDDING_MODEL = "nomic-embed-text"
BASE_URL = "http://localhost:11434"

LLM_MODEL = "qwen3"

# ollama pull nomic-embed_text
embeddings= OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=BASE_URL)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL)

# ### Extract Filters and Ranking Keywords

def extract_filters(user_query:str):

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
                "Tesla profitability" -> {{"company_name": "tesla"}}

                Extract metadata:
                """
    
    metadata = llm_structured.invoke(prompt)
    filters = metadata.model_dump(exclude_none=True)

    return filters


# ### Generate Ranking Keywords

def generate_ranking_keywords(user_query: str):
    # ALT + Z
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

# ### Search the Doc from Vector DB

def build_search_kwargs(filters, ranking_keywords, k=3):

    search_kwargs = {"k": k, 'fetch_k': k*20}

    if filters:
        if len(filters) == 1:
            search_kwargs['filter'] = filters # filters = {'company_name': 'amazon'}

        else:
            # filters = {'company_name': 'amazon', 'doc_type':'10-k'}
            # [{'company_name': 'amazon'}, {'doc_type':'10-k'}]
            filters_conditions = [{k:v} for k, v in filters.items()]
            search_kwargs['filter'] = {"$and": filters_conditions}

    # Add document content filters using ranking keywords
    if ranking_keywords:
        if len(ranking_keywords) == 1:
            search_kwargs['where_document'] = {'$contains': ranking_keywords[0]}
        else:
            search_kwargs['where_document'] = {
                "$or": [{'$contains': keyword} for keyword in ranking_keywords]
            }

    return search_kwargs

def search_docs(query, filters={}, ranking_keywords=[], k=3):
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
    search_kwargs = build_search_kwargs(filters, ranking_keywords, k)

    retriever = vector_store.as_retriever(
        search_type= "mmr",
        search_kwargs = search_kwargs
    )

    return retriever.invoke(query)



# ### Extract Documents Headings and Sub-Headings for ReRanking

def extract_headings_with_content(text):
    """
    Extract markdown headings with one paragraph of content after them.
    
    Args:
        text: Document text content
    
    Returns:
        List of extracted heading + content chunks
    """

    chunks = []

    sections = text.split('\n\n')

    i = 0
    while i< len(sections):
        section = sections[i].strip()

        pattern = r"^#+\s+"
        if re.match(pattern, section):
            heading = section

            # Get the next paragraph/content after the heading
            if i+1 < len(sections):
                next_content = sections[i+1].strip()

                chunk = f"{heading}\n\n{next_content}"
                i = i + 2
            
            else:
                chunk = heading
                i = i + 1

            chunks.append(chunk)

        else:
            i = i + 1

    return chunks


# ### Rank documents using BM25Plus
# https://pypi.org/project/rank-bm25/
def rank_documents_by_keywords(docs, keywords, k=5):
    # 5 -> 20 retriever -> mmr 20*20
    """
    Rank documents using BM25Plus on heading+content chunks.
    
    Args:
        docs: List of Document objects to rank
        keywords: List of keywords to rank by
        k: Number of top documents to return
    
    Returns:
        List of top-k Document objects sorted by BM25 score
    """

    if not docs or not keywords:
        print("Either No doc or keywords found!")
        return docs
    
    query_tokens = " ".join(keywords).lower().split(" ")

    # extract chunks for each document
    doc_chunks = []
    for doc in docs:
        chunks = extract_headings_with_content(doc.page_content)

        combined = " ".join(chunks) if chunks else doc.page_content

        doc_chunks.append(combined.lower().split(' '))

    # Rank using BM25Plust

    bm25 = BM25Plus(doc_chunks)
    doc_scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)

    for rank, idx in enumerate(ranked_indices[:k], 1):
        print(f"   [{rank}] Doc {idx}: score={doc_scores[idx]:.4f}")


    return [docs[i] for i in ranked_indices[:k]]


