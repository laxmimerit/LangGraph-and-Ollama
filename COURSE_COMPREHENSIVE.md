# Private Agentic RAG with LangGraph and Ollama
## Comprehensive Udemy Course Documentation - Advanced Level

**Course Focus:** Building production-ready Retrieval-Augmented Generation (RAG) systems with LangGraph orchestration and local Ollama models for privacy-preserving AI applications.

**Target Audience:** Advanced developers, ML engineers, and AI practitioners with Python knowledge

**Prerequisites:** Python 3.8+, basic understanding of LLMs, vector databases, and machine learning concepts

---

## Course Overview

This advanced course teaches cutting-edge RAG architectures using **LangGraph** for workflow orchestration and **Ollama** for private, local LLM deployment. Students master multiple RAG patterns—from basic retrieval to advanced self-correcting agents—applied to real-world financial document analysis using SEC filings.

### Core Technologies Stack
- **LangGraph 0.2+**: State machine orchestration, conditional routing, agent workflows
- **Ollama**: Local LLM deployment (Qwen3, GPT-OSS, Llama3.2 models)
- **ChromaDB**: Vector database with advanced metadata filtering
- **Docling**: PDF processing with OCR, table extraction, GPU acceleration
- **BM25Plus**: Probabilistic re-ranking for improved relevance
- **Pydantic v2**: Type-safe structured outputs and validation
- **LangChain**: Tool abstractions, document loaders, retrievers
- **Python 3.8+**: TypedDict, Annotated types, operator module

### Real-World Application
**Financial Document Analysis Platform**: Process and analyze SEC filings (10-K annual reports, 10-Q quarterly reports, 8-K current reports) with intelligent retrieval, metadata extraction, and comparative analysis capabilities.

**Dataset:** 1,270+ pages from Amazon, Google, Apple, Microsoft financial filings (2022-2024)

---

# MODULE 1: Ollama Setup and Configuration

## Chapter 1: Local LLM Deployment with Ollama

### Learning Objectives
- Install and configure Ollama for private LLM deployment
- Manage model lifecycle (pull, list, remove, copy)
- Work with Ollama CLI and REST API endpoints
- Create custom models using Modelfiles
- Understand temperature, top_p, top_k parameters

### 1.1 Why Ollama for Private RAG?

**Privacy Benefits:**
- All inference runs on-premise
- No data sent to external APIs
- Full control over model behavior
- Compliance with data regulations (GDPR, HIPAA)

**Cost Benefits:**
- Zero API costs after setup
- Unlimited requests
- Predictable infrastructure costs

**Performance:**
- GPU acceleration (CUDA support)
- Lower latency (no network calls)
- Batch processing capabilities

### 1.2 Ollama CLI Commands

#### Model Management
```bash
# Download models from Ollama registry
ollama pull qwen3                    # Primary LLM (general-purpose)
ollama pull gpt-oss                  # LLM with reasoning capabilities
ollama pull llama3.2                 # Alternative LLM
ollama pull qwen3:0.6b               # Lightweight variant
ollama pull nomic-embed-text         # Embedding model (384 dimensions)

# List installed models
ollama list
# Output: NAME              ID         SIZE    MODIFIED
#         qwen3:latest      abc123     4.9GB   2 days ago

# Show running processes
ollama ps
# Shows currently loaded models and memory usage

# Remove models to free space
ollama rm model_name

# Copy model (useful for customization)
ollama cp qwen3 financial-analyzer
```

#### Model Execution
```bash
# Start Ollama service (runs on port 11434)
ollama serve

# Run model interactively
ollama run qwen3

# Inside model - session commands
/set temperature 0.7      # Adjust randomness
/show info                # Display model details
/load session_name        # Load saved session
/save session_name        # Save current context
/clear                    # Clear conversation history
/bye                      # Exit
"""                       # Start multi-line input
Your multi-line
message here
"""
```

### 1.3 Ollama REST API

**Base URL:** `http://localhost:11434`

#### Generate Endpoint (Completion)
```python
import requests

response = requests.post('http://localhost:11434/api/generate', json={
    "model": "qwen3",
    "prompt": "Explain PageRAG in one sentence",
    "stream": False,
    "options": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40
    }
})

print(response.json()['response'])
```

#### Chat Endpoint (Conversational)
```python
response = requests.post('http://localhost:11434/api/chat', json={
    "model": "qwen3",
    "messages": [
        {"role": "system", "content": "You are a financial analyst"},
        {"role": "user", "content": "What is Amazon's revenue?"}
    ],
    "stream": False
})
```

#### Embeddings Endpoint
```python
response = requests.post('http://localhost:11434/api/embed', json={
    "model": "nomic-embed-text",
    "input": "Financial document analysis with RAG"
})

embeddings = response.json()['embeddings']  # 384-dimensional vector
```

### 1.4 Custom Models with Modelfiles

**Use Case:** Create specialized financial analyst model

**Modelfile:** `financial_analyst.modelfile`
```dockerfile
FROM qwen3

# Model parameters
PARAMETER temperature 0.3           # Lower for factual accuracy
PARAMETER top_p 0.85
PARAMETER top_k 30
PARAMETER num_ctx 4096             # Context window size

# System prompt
SYSTEM """You are an expert financial analyst specializing in SEC filings (10-K, 10-Q, 8-K).

Your responsibilities:
1. Extract financial metrics accurately
2. Provide comparative analysis across companies
3. Always cite sources: (Company: X, Year: Y, Quarter: Z, Page: N)
4. Use markdown tables for comparisons
5. Explain financial terms clearly

Guidelines:
- Prioritize accuracy over speed
- Reference specific document sections
- Flag data inconsistencies
- Provide context for metrics"""

# Custom message template
TEMPLATE """### System:
{{ .System }}

### User:
{{ .Prompt }}

### Assistant:
"""
```

**Create and use custom model:**
```bash
ollama create financial-analyst -f financial_analyst.modelfile
ollama run financial-analyst
```

### 1.5 LangChain Integration

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings

# LLM initialization
llm = ChatOllama(
    model="qwen3",
    base_url="http://localhost:11434",
    temperature=0.7,
    num_ctx=4096
)

# Embeddings initialization
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Test invocation
response = llm.invoke("What is RAG?")
print(response.content)

# Generate embeddings
vector = embeddings.embed_query("financial document")
print(f"Embedding dimensions: {len(vector)}")  # 384
```

### Key Takeaways
- Ollama provides privacy-first LLM deployment
- Models run entirely on local infrastructure
- CLI and API provide flexible access patterns
- Modelfiles enable custom model creation
- LangChain integration simplifies development

---

# MODULE 2: LangGraph Fundamentals

## Chapter 2: State Machines and Workflow Orchestration

### Learning Objectives
- Understand LangGraph's state machine paradigm
- Define typed states with TypedDict
- Create nodes (processing functions)
- Connect nodes with edges (linear and conditional)
- Build and compile executable graphs
- Handle state updates and merging

### 2.1 What is LangGraph?

**LangGraph** is a low-level orchestration framework for building stateful, multi-step agent workflows. Unlike sequential pipelines, LangGraph enables:

- **Conditional Routing**: Dynamic flow based on runtime decisions
- **State Management**: Shared memory accessible by all nodes
- **Cyclic Workflows**: Loops for iterative refinement
- **Human-in-the-Loop**: Interrupt points for manual intervention
- **Checkpointing**: Save/resume execution state

**Use Cases:**
- Multi-step agents (ReAct pattern)
- Self-correcting workflows (Reflexion, CRAG)
- Multi-agent systems
- Complex RAG architectures

### 2.2 Core Concepts

#### State (TypedDict)
**State** is a typed dictionary defining shared memory structure accessible by all nodes.

```python
from typing_extensions import TypedDict

class SimpleState(TypedDict):
    input_text: str      # User input
    output_text: str     # Processed output
```

**Characteristics:**
- Type-safe schema
- Immutable structure
- Nodes read from and write to state
- LangGraph merges updates automatically

#### Nodes (Functions)
**Nodes** are pure functions that process data and return state updates.

```python
def process_input(state: SimpleState) -> dict:
    """Convert input to uppercase"""
    output_text = state['input_text'].upper()
    return {'output_text': output_text}

def add_prefix(state: SimpleState) -> dict:
    """Add greeting prefix"""
    output = "Hey, " + state['output_text']
    return {'output_text': output}

def add_suffix(state: SimpleState) -> dict:
    """Add exclamation suffix"""
    output = state['output_text'] + "!"
    return {'output_text': output}
```

**Node Characteristics:**
- Take state as input
- Return partial state updates (dict)
- LangGraph merges updates into full state
- Pure functions (no side effects recommended)

#### Edges (Connections)
**Edges** define workflow between nodes.

**Linear Edges:**
```python
builder.add_edge("node_a", "node_b")  # Always go from A to B
builder.add_edge(START, "first_node") # Entry point
builder.add_edge("last_node", END)    # Exit point
```

**Conditional Edges:**
```python
def router_function(state):
    if condition:
        return "path_a"
    else:
        return "path_b"

builder.add_conditional_edges(
    "decision_node",
    router_function,
    {
        "path_a": "node_for_a",
        "path_b": "node_for_b"
    }
)
```

### 2.3 Building Your First Graph

**Complete Example: Text Processing Pipeline**

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Define State
class SimpleState(TypedDict):
    input_text: str
    output_text: str

# 2. Define Nodes
def process_input(state: SimpleState) -> dict:
    output = state['input_text'].upper()
    return {'output_text': output}

def add_prefix(state: SimpleState) -> dict:
    print(f"[PREFIX NODE] Current state: {state}")
    output = "Hey, " + state['output_text']
    return {'output_text': output}

def add_suffix(state: SimpleState) -> dict:
    print(f"[SUFFIX NODE] Current state: {state}")
    output = state['output_text'] + "!"
    return {'output_text': output}

# 3. Build Graph
def create_simple_graph():
    builder = StateGraph(SimpleState)

    # Add nodes
    builder.add_node("process_input", process_input)
    builder.add_node("add_prefix", add_prefix)
    builder.add_node("add_suffix", add_suffix)

    # Connect nodes (linear flow)
    builder.add_edge(START, "process_input")
    builder.add_edge("process_input", "add_prefix")
    builder.add_edge("add_prefix", "add_suffix")
    builder.add_edge("add_suffix", END)

    # Compile to executable graph
    graph = builder.compile()
    return graph

# 4. Execute
graph = create_simple_graph()
result = graph.invoke({'input_text': "hello"})

print(result)
# Output: {'input_text': 'hello', 'output_text': 'Hey, HELLO!'}
```

**Execution Flow:**
```
START → process_input (hello → HELLO)
      → add_prefix (HELLO → Hey, HELLO)
      → add_suffix (Hey, HELLO → Hey, HELLO!)
      → END
```

### 2.4 State Merging Behavior

**How LangGraph Merges Updates:**

```python
# Initial state
state = {'input_text': 'hello', 'output_text': ''}

# Node 1 returns partial update
update_1 = {'output_text': 'HELLO'}
# State becomes: {'input_text': 'hello', 'output_text': 'HELLO'}

# Node 2 returns partial update
update_2 = {'output_text': 'Hey, HELLO'}
# State becomes: {'input_text': 'hello', 'output_text': 'Hey, HELLO'}

# Final state preserves all fields
# {'input_text': 'hello', 'output_text': 'Hey, HELLO!'}
```

**Important:** Node return values override state fields, not merge into them (unless using reducers).

### 2.5 Advanced State Pattern: Message Accumulation

```python
from typing import Annotated
import operator

class MessageState(TypedDict):
    messages: Annotated[list, operator.add]  # Accumulator pattern
```

**Behavior:**
```python
# Initial state
state = {'messages': []}

# Node 1 returns
update_1 = {'messages': [HumanMessage("Hello")]}
# State: {'messages': [HumanMessage("Hello")]}

# Node 2 returns
update_2 = {'messages': [AIMessage("Hi there")]}
# State: {'messages': [HumanMessage("Hello"), AIMessage("Hi there")]}
```

The `Annotated[list, operator.add]` tells LangGraph to **append** new messages instead of replacing the entire list.

### 2.6 Visualizing Graphs

```python
from IPython.display import Image, display

# Generate Mermaid diagram
display(Image(graph.get_graph().draw_mermaid_png()))
```

**Output:**
```
START → process_input → add_prefix → add_suffix → END
```

### Key Takeaways
- LangGraph provides state machine abstraction for workflows
- State (TypedDict) = shared memory
- Nodes (functions) = processing units
- Edges = workflow connections
- `Annotated[list, operator.add]` = accumulator pattern
- Compile graph before execution

---

# MODULE 3: RAG Applications - Core Chapter

## Chapter 3: PageRAG - Data Ingestion Pipeline

### Learning Objectives
- Extract text from PDFs page-by-page using Docling
- Generate metadata from filenames and LLM analysis
- Implement file deduplication with SHA-256 hashing
- Store documents in ChromaDB with rich metadata
- Design scalable ingestion pipelines for financial documents

### 3.1 Architecture Overview

**PageRAG Data Ingestion Pipeline:**
```
PDF Files → Metadata Extraction → Page Extraction → Deduplication Check
         → Vector Embedding → ChromaDB Storage
```

**Key Innovation:** Page-wise ingestion (not document-wise) enables granular retrieval and precise citations.

**Tech Stack:**
- **Docling**: PDF → Markdown with OCR, table extraction
- **ChromaDB**: Vector storage with metadata filtering
- **Pydantic**: Metadata schema validation
- **SHA-256**: File hashing for deduplication

### 3.2 Metadata Schema Design

**File:** `scripts/schemas.py`

```python
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

# Enum for strict type validation
class DocType(Enum):
    TEN_K = "10-k"      # Annual report
    TEN_Q = "10-q"      # Quarterly report
    EIGHT_K = "8-k"     # Current report
    OTHER = "other"

class FiscalQuarter(Enum):
    Q1 = "q1"
    Q2 = "q2"
    Q3 = "q3"
    Q4 = "q4"

# Metadata model with validation
class ChunkMetadata(BaseModel):
    company_name: Optional[str] = Field(
        default=None,
        description="Company name (lowercase: 'amazon', 'google', 'apple')"
    )
    doc_type: Optional[DocType] = Field(
        default=None,
        description="Document type (10-k, 10-q, 8-k)"
    )
    fiscal_year: Optional[int] = Field(
        default=None,
        ge=1950,  # Greater than or equal to
        le=2050,  # Less than or equal to
        description="Fiscal year"
    )
    fiscal_quarter: Optional[FiscalQuarter] = Field(
        default=None,
        description="Fiscal quarter (q1-q4)"
    )

    # Serialize enums as values
    model_config = {"use_enum_values": True}

# Usage
metadata = ChunkMetadata(
    company_name="amazon",
    doc_type=DocType.TEN_K,
    fiscal_year=2023
)

# Convert to dict for ChromaDB
filters = metadata.model_dump(exclude_none=True)
# {'company_name': 'amazon', 'doc_type': '10-k', 'fiscal_year': 2023}
```

**Benefits:**
- Type safety with Enums
- Automatic validation (year range, required fields)
- Clean serialization for databases
- Self-documenting schema

### 3.3 PDF Processing with Docling

**Docling** converts PDFs to markdown with:
- OCR support (handles scanned documents)
- Table extraction (preserves structure)
- Image description (optional)
- GPU acceleration (CUDA)
- Page break placeholders

```python
from docling.document_converter import DocumentConverter

def extract_pdf_pages(pdf_path: str) -> list[str]:
    """
    Extract text from PDF, split by pages.

    Returns:
        List of page contents (markdown format)
    """
    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    # Export to markdown with page break markers
    page_break = "<!-- page break -->"
    markdown_text = result.document.export_to_markdown(
        page_break_placeholder=page_break
    )

    # Split into individual pages
    pages = markdown_text.split(page_break)

    return pages

# Usage
pages = extract_pdf_pages('data/amazon/amazon 10-k 2023.pdf')
print(f"Extracted {len(pages)} pages")  # 95 pages
print(pages[0][:200])  # First 200 characters of page 1
```

**Output Example:**
```markdown
## UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

### FORM 10-K

☑ ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934

For the fiscal year ended December 31, 2023

**Commission File Number**: 0-51994

### AMAZON.COM, INC.
...
```

### 3.4 Filename-Based Metadata Extraction

**Expected Filename Format:**
```
{company} {doc_type} [{quarter}] {year}.pdf

Examples:
- amazon 10-k 2023.pdf
- google 10-q q1 2024.pdf
- apple 8-k 2024.pdf
```

**Extraction Function:**
```python
def extract_metadata_from_filename(filename: str) -> dict:
    """
    Extract metadata from structured filename.

    Args:
        filename: PDF filename

    Returns:
        Dict with company_name, doc_type, fiscal_year, fiscal_quarter
    """
    name = filename.replace('.pdf', '')
    parts = name.split()

    metadata = {}

    # Determine if quarterly (4 parts) or annual (3 parts)
    if len(parts) == 4:
        # Quarterly: amazon 10-q q1 2024
        metadata['company_name'] = parts[0]
        metadata['doc_type'] = parts[1]
        metadata['fiscal_quarter'] = parts[2]  # q1, q2, q3, q4
        metadata['fiscal_year'] = int(parts[3])
    else:
        # Annual: amazon 10-k 2023
        metadata['company_name'] = parts[0]
        metadata['doc_type'] = parts[1]
        metadata['fiscal_quarter'] = None
        metadata['fiscal_year'] = int(parts[2])

    return metadata

# Examples
extract_metadata_from_filename('amazon 10-k 2023.pdf')
# {'company_name': 'amazon', 'doc_type': '10-k', 'fiscal_quarter': None, 'fiscal_year': 2023}

extract_metadata_from_filename('google 10-q q1 2024.pdf')
# {'company_name': 'google', 'doc_type': '10-q', 'fiscal_quarter': 'q1', 'fiscal_year': 2024}
```

### 3.5 File Deduplication with Hashing

**Problem:** Prevent duplicate ingestion of same files (renamed copies)

**Solution:** SHA-256 hash of file content

```python
import hashlib

def compute_file_hash(file_path: str) -> str:
    """
    Compute SHA-256 hash of file content.

    Returns:
        64-character hex hash
    """
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read in chunks for large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()

# Usage
hash1 = compute_file_hash('amazon 10-k 2023.pdf')
hash2 = compute_file_hash('amazon 10-k 2023 copy.pdf')

print(hash1 == hash2)  # True - identical content
# 'c08079bc14250c896f3ca151f9a72ecc1ddcb9ca8e5b021539e91af10fae5c4b'
```

**Check for Processed Files:**
```python
# Get existing file hashes from ChromaDB
existing_docs = vector_store.get(
    where={"file_hash": {"$ne": ""}},  # Not empty
    include=['metadatas']
)

processed_hashes = {
    m.get('file_hash')
    for m in existing_docs['metadatas']
    if m.get('file_hash')
}

# Check if file already processed
file_hash = compute_file_hash(pdf_path)
if file_hash in processed_hashes:
    print(f"[SKIP] Already processed: {pdf_path}")
    return
```

### 3.6 ChromaDB Vector Store Setup

```python
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Configuration
CHROMA_DIR = "./chroma_financial_db"
COLLECTION_NAME = "financial_docs"
EMBEDDING_MODEL = "nomic-embed-text"
BASE_URL = "http://localhost:11434"

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

# Check collection size
count = vector_store._collection.count()
print(f"Collection contains {count} documents")  # 1,270 pages
```

### 3.7 Complete Ingestion Pipeline

```python
from pathlib import Path
from langchain_core.documents import Document

def ingest_docs_in_vectordb(pdf_path: Path):
    """
    Ingest PDF into vector database with page-wise chunking.

    Steps:
    1. Check if already processed (hash)
    2. Extract pages from PDF
    3. Extract metadata from filename
    4. Create Document for each page
    5. Add to vector store
    """
    print(f"[INGEST] Processing: {pdf_path.name}")

    # Step 1: Deduplication check
    file_hash = compute_file_hash(pdf_path)
    if file_hash in processed_hashes:
        print(f"[SKIP] Already processed")
        return

    # Step 2: Extract pages
    pages = extract_pdf_pages(pdf_path)
    print(f"[EXTRACT] Found {len(pages)} pages")

    # Step 3: Extract file-level metadata
    file_metadata = extract_metadata_from_filename(pdf_path.name)

    # Step 4: Create documents for each page
    documents = []
    for page_num, page_text in enumerate(pages, start=1):
        # Combine file metadata + page metadata
        metadata_dict = file_metadata.copy()
        metadata_dict['page'] = page_num
        metadata_dict['file_hash'] = file_hash
        metadata_dict['source_file'] = pdf_path.name

        doc = Document(
            page_content=page_text,
            metadata=metadata_dict
        )
        documents.append(doc)

    # Step 5: Add to vector store
    vector_store.add_documents(documents=documents)
    print(f"[SUCCESS] Added {len(documents)} pages to vector store")

# Batch processing
data_path = Path("data")
pdf_files = list(data_path.rglob("*.pdf"))  # Recursive search

for pdf_path in pdf_files:
    ingest_docs_in_vectordb(pdf_path)

print(f"[COMPLETE] Total documents: {vector_store._collection.count()}")
```

**Sample Metadata in ChromaDB:**
```python
vector_store.get(
    where={"company_name": "amazon"},
    limit=1
)
```

**Output:**
```python
{
    'ids': ['doc_123'],
    'metadatas': [{
        'company_name': 'amazon',
        'doc_type': '10-k',
        'fiscal_year': 2023,
        'fiscal_quarter': None,
        'page': 24,
        'file_hash': 'c08079bc1425...',
        'source_file': 'amazon 10-k 2023.pdf'
    }],
    'documents': ['## CONSOLIDATED STATEMENTS OF OPERATIONS\n...'],
    'embeddings': None  # Not returned by default
}
```

### 3.8 Production Considerations

**Scalability:**
- Batch processing with progress bars (tqdm)
- Parallel PDF processing (multiprocessing)
- Chunk size optimization (default: page-wise)

**Error Handling:**
```python
try:
    pages = extract_pdf_pages(pdf_path)
except Exception as e:
    print(f"[ERROR] Failed to process {pdf_path}: {e}")
    with open('failed_files.log', 'a') as f:
        f.write(f"{pdf_path}\n")
    return
```

**Monitoring:**
```python
import time

start_time = time.time()
ingest_docs_in_vectordb(pdf_path)
elapsed = time.time() - start_time

print(f"[METRICS] Processed {pdf_path.name} in {elapsed:.2f}s")
```

### Key Takeaways
- Page-wise ingestion enables precise citations
- Metadata extraction (filename + LLM) improves filtering
- SHA-256 hashing prevents duplicate ingestion
- ChromaDB provides efficient vector storage with metadata filtering
- Docling handles complex PDF layouts with OCR
- Pydantic ensures metadata consistency

**Next:** Learn to retrieve and re-rank these documents with BM25Plus

---

## Chapter 4: Data Retrieval and BM25Plus Re-Ranking

### Learning Objectives
- Extract metadata filters from natural language queries using LLMs
- Generate SEC filing-specific ranking keywords
- Build complex ChromaDB filters (AND/OR logic)
- Implement MMR (Maximal Marginal Relevance) retrieval
- Re-rank results with BM25Plus algorithm
- Extract heading+content chunks for better ranking

### 4.1 Retrieval Architecture

**Two-Stage Retrieval Pipeline:**
```
User Query → 1. Metadata Extraction (LLM)
           → 2. Keyword Generation (LLM)
           → 3. Vector Search (ChromaDB + MMR)
           → 4. BM25+ Re-Ranking
           → Top-K Results
```

**Why Two Stages?**
1. **Vector Search**: Fast initial retrieval (60+ candidates)
2. **Re-Ranking**: Precision refinement with BM25Plus on document structure

### 4.2 Metadata Filter Extraction

**File:** `scripts/utils.py`

**Goal:** Convert natural language to structured filters

```python
from langchain_ollama import ChatOllama
from scripts.schemas import ChunkMetadata

llm = ChatOllama(model="qwen3", base_url="http://localhost:11434")

def extract_filters(user_query: str) -> dict:
    """
    Extract metadata filters from natural language query.

    Args:
        user_query: "Amazon's Q3 2024 revenue"

    Returns:
        {'company_name': 'amazon', 'doc_type': '10-q',
         'fiscal_year': 2024, 'fiscal_quarter': 'q3'}
    """
    llm_structured = llm.with_structured_output(ChunkMetadata)

    prompt = f"""Extract metadata filters from the query.
Return None for fields not mentioned.

USER QUERY: {user_query}

COMPANY MAPPINGS:
- Amazon/AMZN → amazon
- Google/Alphabet/GOOGL/GOOG → google
- Apple/AAPL → apple
- Microsoft/MSFT → microsoft
- Tesla/TSLA → tesla
- Nvidia/NVDA → nvidia
- Meta/Facebook/FB → meta

DOC TYPE MAPPINGS:
- Annual report → 10-k
- Quarterly report → 10-q
- Current report → 8-k

EXAMPLES:
"Amazon Q3 2024 revenue" →
{{"company_name": "amazon", "doc_type": "10-q", "fiscal_year": 2024, "fiscal_quarter": "q3"}}

"Apple 2023 annual report" →
{{"company_name": "apple", "doc_type": "10-k", "fiscal_year": 2023}}

"Tesla profitability" →
{{"company_name": "tesla"}}

Extract metadata:"""

    metadata = llm_structured.invoke(prompt)
    filters = metadata.model_dump(exclude_none=True)

    return filters

# Examples
extract_filters("Amazon's cash flow in 2024")
# {'company_name': 'amazon', 'fiscal_year': 2024}

extract_filters("Google Q1 2024 earnings")
# {'company_name': 'google', 'doc_type': '10-q', 'fiscal_year': 2024, 'fiscal_quarter': 'q1'}
```

**Key Feature:** `with_structured_output(ChunkMetadata)` forces LLM to return valid Pydantic model.

### 4.3 SEC Filing Keyword Generation

**Goal:** Generate exact terms from 10-K/10-Q filings for better ranking

```python
from scripts.schemas import RankingKeywords

class RankingKeywords(BaseModel):
    keywords: List[str] = Field(
        ...,
        description="Exactly 5 financial keywords",
        min_length=5,
        max_length=5
    )

def generate_ranking_keywords(user_query: str) -> list[str]:
    """
    Generate EXACTLY 5 SEC filing-specific keywords.

    Args:
        user_query: "Amazon's revenue in 2023"

    Returns:
        ["revenue", "net revenue", "consolidated statements of operations",
         "total revenue", "net sales"]
    """
    prompt = f"""Generate EXACTLY 5 financial keywords from SEC filings terminology.

USER QUERY: {user_query}

USE EXACT TERMS FROM 10-K/10-Q FILINGS:

STATEMENT HEADINGS:
"consolidated statements of operations"
"consolidated balance sheets"
"consolidated statements of cash flows"
"consolidated statements of stockholders equity"

INCOME STATEMENT:
"revenue", "net revenue", "cost of revenue", "gross profit"
"operating income", "net income", "earnings per share"

BALANCE SHEET:
"total assets", "cash and cash equivalents", "total liabilities"
"stockholders equity", "working capital", "long-term debt"

CASH FLOWS:
"cash flows from operating activities"
"net cash provided by operating activities"
"cash flows from investing activities"
"free cash flow", "capital expenditures"

RULES:
- Return EXACTLY 5 keywords
- Use exact phrases from SEC filings
- Match query topic (revenue → revenue terms)
- Use plural forms: "cash flows", "stockholders equity"

EXAMPLES:
"revenue analysis" →
["revenue", "net revenue", "total revenue",
 "consolidated statements of operations", "net sales"]

"cash flow performance" →
["consolidated statements of cash flows",
 "cash flows from operating activities",
 "net cash provided by operating activities",
 "free cash flow", "operating activities"]

Generate EXACTLY 5 keywords:"""

    llm_structured = llm.with_structured_output(RankingKeywords)
    result = llm_structured.invoke(prompt)

    return result.keywords

# Example
generate_ranking_keywords("Amazon's profitability in 2023")
# ['net income', 'operating income', 'gross profit',
#  'earnings per share', 'consolidated statements of operations']
```

**Why SEC-Specific Keywords?**
- Financial documents use standardized terminology
- "Revenue" appears hundreds of times → need specific context
- "Consolidated statements of operations" pinpoints exact section
- Improves BM25 ranking precision

### 4.4 Building ChromaDB Filters

**Goal:** Combine metadata filters with content filters

```python
def build_search_kwargs(filters: dict, ranking_keywords: list, k: int = 3) -> dict:
    """
    Build search_kwargs for ChromaDB retriever.

    Args:
        filters: {'company_name': 'amazon', 'fiscal_year': 2023}
        ranking_keywords: ['revenue', 'net income', ...]
        k: Final number of documents

    Returns:
        {
            'k': 3,
            'fetch_k': 60,  # For MMR diversity
            'filter': {'$and': [...]},  # Metadata filters
            'where_document': {'$or': [...]}  # Content filters
        }
    """
    search_kwargs = {
        'k': k,
        'fetch_k': k * 20  # Fetch 20x more for MMR
    }

    # Metadata filters (AND logic)
    if filters:
        if len(filters) == 1:
            # Single filter: {'company_name': 'amazon'}
            search_kwargs['filter'] = filters
        else:
            # Multiple filters: AND them together
            # {'company_name': 'amazon', 'fiscal_year': 2023}
            # → {"$and": [{'company_name': 'amazon'}, {'fiscal_year': 2023}]}
            filters_conditions = [{k: v} for k, v in filters.items()]
            search_kwargs['filter'] = {"$and": filters_conditions}

    # Content filters (OR logic)
    if ranking_keywords:
        if len(ranking_keywords) == 1:
            search_kwargs['where_document'] = {'$contains': ranking_keywords[0]}
        else:
            # Document must contain AT LEAST ONE keyword
            search_kwargs['where_document'] = {
                "$or": [
                    {'$contains': keyword}
                    for keyword in ranking_keywords
                ]
            }

    return search_kwargs

# Example
build_search_kwargs(
    filters={'company_name': 'amazon', 'fiscal_year': 2023},
    ranking_keywords=['revenue', 'net income'],
    k=3
)
```

**Output:**
```python
{
    'k': 3,
    'fetch_k': 60,
    'filter': {
        "$and": [
            {'company_name': 'amazon'},
            {'fiscal_year': 2023}
        ]
    },
    'where_document': {
        "$or": [
            {'$contains': 'revenue'},
            {'$contains': 'net income'}
        ]
    }
}
```

**ChromaDB Filter Syntax:**
- `$and`: All conditions must match
- `$or`: At least one condition must match
- `$contains`: Document content contains keyword
- `$ne`: Not equal
- `$gt`, `$lt`: Greater/less than (for numeric fields)

### 4.5 MMR Retrieval Strategy

**MMR (Maximal Marginal Relevance)** balances relevance and diversity.

```python
def search_docs(query: str, filters: dict = {},
                ranking_keywords: list = [], k: int = 3) -> list:
    """
    Search documents with metadata and content filters.

    Args:
        query: "Amazon's revenue in 2023"
        filters: {'company_name': 'amazon', 'fiscal_year': 2023}
        ranking_keywords: ['revenue', 'net revenue', ...]
        k: Number of final results

    Returns:
        List of Document objects
    """
    search_kwargs = build_search_kwargs(filters, ranking_keywords, k)

    retriever = vector_store.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance
        search_kwargs=search_kwargs
    )

    return retriever.invoke(query)

# Example
docs = search_docs(
    query="Amazon's revenue in 2023",
    filters={'company_name': 'amazon', 'fiscal_year': 2023},
    ranking_keywords=['revenue', 'net revenue', 'total revenue'],
    k=5
)
```

**MMR Parameters:**
- `k=5`: Return 5 final documents
- `fetch_k=100`: Fetch 100 candidates first
- **MMR Algorithm:**
  1. Fetch top 100 by similarity
  2. Select most relevant
  3. Select next that's relevant BUT different from #2
  4. Repeat until 5 documents

**vs. Similarity Search:**
- Similarity: All 5 might be from same page/section
- MMR: 5 diverse pages covering different aspects

### 4.6 Heading Extraction for Re-Ranking

**Goal:** Extract markdown headings with following paragraph for BM25

```python
import re

def extract_headings_with_content(text: str) -> list[str]:
    """
    Extract markdown headings with one paragraph after each.

    Input:
        ## Revenue Analysis

        Amazon's revenue grew by 12%...

        ## Cost Structure

        Operating costs increased by...

    Returns:
        [
            "## Revenue Analysis\n\nAmazon's revenue grew by 12%...",
            "## Cost Structure\n\nOperating costs increased by..."
        ]
    """
    chunks = []
    sections = text.split('\n\n')  # Split by double newline

    i = 0
    while i < len(sections):
        section = sections[i].strip()

        # Check if section is a markdown heading
        heading_pattern = r"^#+\s+"  # Matches ##, ###, etc.
        if re.match(heading_pattern, section):
            heading = section

            # Get next paragraph after heading
            if i + 1 < len(sections):
                next_content = sections[i + 1].strip()
                chunk = f"{heading}\n\n{next_content}"
                i += 2
            else:
                chunk = heading
                i += 1

            chunks.append(chunk)
        else:
            i += 1

    return chunks

# Example
text = """## CONSOLIDATED STATEMENTS OF OPERATIONS

(In millions, except per share data)

| Year Ended December 31, | 2023 | 2022 | 2021 |
|-------------------------|------|------|------|
| Net revenue            | $574,785 | $513,983 | $469,822 |

## Operating Expenses

Costs increased due to..."""

chunks = extract_headings_with_content(text)
# [
#   "## CONSOLIDATED STATEMENTS OF OPERATIONS\n\n(In millions, except per share data)...",
#   "## Operating Expenses\n\nCosts increased due to..."
# ]
```

**Why Heading+Content?**
- Headings provide context ("Revenue" vs. "Revenue Analysis")
- Content provides substance for ranking
- Balances structure and detail for BM25

### 4.7 BM25Plus Re-Ranking Algorithm

**BM25Plus** is a probabilistic ranking function that scores documents based on term frequency.

```python
from rank_bm25 import BM25Plus

def rank_documents_by_keywords(docs: list, keywords: list, k: int = 5) -> list:
    """
    Re-rank documents using BM25Plus on heading+content chunks.

    Args:
        docs: List of Document objects from initial retrieval
        keywords: Ranking keywords from LLM
        k: Number of top documents to return

    Returns:
        Top-k documents sorted by BM25 score
    """
    if not docs or not keywords:
        print("[WARN] No docs or keywords found")
        return docs

    # Tokenize query keywords
    query_tokens = " ".join(keywords).lower().split()

    # Extract heading+content chunks from each document
    doc_chunks = []
    for doc in docs:
        chunks = extract_headings_with_content(doc.page_content)

        # Combine chunks or use full content
        combined = " ".join(chunks) if chunks else doc.page_content

        # Tokenize document
        doc_chunks.append(combined.lower().split())

    # Initialize BM25Plus
    bm25 = BM25Plus(doc_chunks)

    # Score documents
    doc_scores = bm25.get_scores(query_tokens)

    # Rank by score (descending)
    ranked_indices = sorted(
        range(len(doc_scores)),
        key=lambda i: doc_scores[i],
        reverse=True
    )

    # Print ranking details
    for rank, idx in enumerate(ranked_indices[:k], 1):
        print(f"   [{rank}] Doc {idx}: score={doc_scores[idx]:.4f}")

    # Return top-k documents
    return [docs[i] for i in ranked_indices[:k]]

# Example
docs = search_docs(
    query="Amazon revenue 2023",
    filters={'company_name': 'amazon', 'fiscal_year': 2023},
    ranking_keywords=['revenue', 'net revenue', 'total revenue'],
    k=10  # Fetch 10 for re-ranking
)

# Re-rank to top 5
ranked_docs = rank_documents_by_keywords(
    docs=docs,
    keywords=['revenue', 'net revenue', 'total revenue'],
    k=5
)

# Output:
#    [1] Doc 18: score=21.5508
#    [2] Doc 11: score=20.9299
#    [3] Doc 9: score=20.6941
#    [4] Doc 1: score=19.3327
#    [5] Doc 10: score=17.7692
```

**BM25Plus Formula (Simplified):**
```
score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl) + delta)
```

Where:
- `f(qi, D)`: Frequency of keyword `qi` in document `D`
- `IDF(qi)`: Inverse document frequency (rare terms score higher)
- `|D|`: Document length
- `avgdl`: Average document length
- `k1, b, delta`: Tuning parameters

**BM25Plus vs. BM25:**
- BM25Plus adds `delta` parameter
- Prevents zero scores for documents without exact keyword matches
- Better handles long documents

### 4.8 Complete Retrieval Function

```python
def retrieve_and_rank(user_query: str, k: int = 5) -> list:
    """
    Complete retrieval pipeline with filter extraction and re-ranking.

    Args:
        user_query: "What was Amazon's revenue in Q2 2024?"
        k: Number of final results

    Returns:
        Top-k ranked Document objects
    """
    print(f"[QUERY] {user_query}")

    # Step 1: Extract metadata filters
    filters = extract_filters(user_query)
    print(f"[FILTERS] {filters}")

    # Step 2: Generate ranking keywords
    keywords = generate_ranking_keywords(user_query)
    print(f"[KEYWORDS] {keywords}")

    # Step 3: Initial retrieval (fetch 10x for re-ranking)
    initial_docs = search_docs(
        query=user_query,
        filters=filters,
        ranking_keywords=keywords,
        k=k * 10  # Fetch more for re-ranking
    )
    print(f"[RETRIEVED] {len(initial_docs)} initial documents")

    # Step 4: Re-rank with BM25Plus
    final_docs = rank_documents_by_keywords(
        docs=initial_docs,
        keywords=keywords,
        k=k
    )
    print(f"[RANKED] {len(final_docs)} final documents")

    return final_docs

# Usage
docs = retrieve_and_rank("Amazon's cash flow in 2023", k=3)

# Output:
# [QUERY] Amazon's cash flow in 2023
# [FILTERS] {'company_name': 'amazon', 'fiscal_year': 2023}
# [KEYWORDS] ['consolidated statements of cash flows', 'cash flows from operating activities',
#             'net cash provided by operating activities', 'free cash flow', 'capital expenditures']
# [RETRIEVED] 30 initial documents
#    [1] Doc 12: score=25.3421
#    [2] Doc 5: score=23.1234
#    [3] Doc 18: score=21.8765
# [RANKED] 3 final documents
```

### 4.9 Production Optimizations

**Caching LLM Calls:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def extract_filters_cached(user_query: str) -> tuple:
    filters = extract_filters(user_query)
    return tuple(sorted(filters.items()))  # Hashable for cache
```

**Batch Processing:**
```python
from langchain.retrievers import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(),
    llm=llm
)
```

**Monitoring:**
```python
import time

def retrieve_and_rank_with_metrics(user_query: str, k: int = 5):
    metrics = {}

    start = time.time()
    filters = extract_filters(user_query)
    metrics['filter_time'] = time.time() - start

    start = time.time()
    keywords = generate_ranking_keywords(user_query)
    metrics['keyword_time'] = time.time() - start

    start = time.time()
    docs = search_docs(user_query, filters, keywords, k * 10)
    metrics['search_time'] = time.time() - start

    start = time.time()
    ranked = rank_documents_by_keywords(docs, keywords, k)
    metrics['rank_time'] = time.time() - start

    print(f"[METRICS] {metrics}")
    return ranked
```

### Key Takeaways
- LLM-powered metadata extraction enables natural language queries
- SEC-specific keywords improve ranking precision
- ChromaDB supports complex AND/OR filter logic
- MMR balances relevance and diversity
- BM25Plus re-ranking refines initial retrieval
- Heading+content chunks provide better ranking signals
- Two-stage retrieval (vector + BM25) outperforms single-stage

**Next:** Build an agentic RAG system with tool-based retrieval

---

## Chapter 5: Agentic PageRAG with LangGraph

### Learning Objectives
- Build LangGraph agents with tool integration
- Implement retrieve_docs as a @tool function
- Design effective system prompts for financial analysis
- Handle multi-document retrieval for comparisons
- Format outputs with markdown, tables, and citations
- Implement conditional routing (agent ↔ tools)

### 5.1 Agentic RAG Architecture

**Traditional RAG:**
```
Query → Retrieve → Generate → Answer
```

**Agentic RAG:**
```
Query → Agent Decides → Call retrieve_docs Tool →
        Agent Analyzes → (Optional: Call tool again) →
        Agent Generates → Answer
```

**Key Differences:**
- **Decision-Making**: Agent chooses when/how to call tools
- **Multi-Turn**: Agent can call tools multiple times
- **Decomposition**: Agent breaks complex queries into sub-questions
- **Reasoning**: Agent explains its process

### 5.2 State Definition

```python
from typing_extensions import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # Message accumulation

# Messages can be:
# - HumanMessage: User input
# - AIMessage: Agent responses
# - ToolMessage: Tool results
```

**Why Annotated[list, operator.add]?**
```python
# Without Annotated
state = {'messages': [HumanMessage("Hi")]}
update = {'messages': [AIMessage("Hello")]}
# Result: {'messages': [AIMessage("Hello")]}  # Replaced!

# With Annotated[list, operator.add]
state = {'messages': [HumanMessage("Hi")]}
update = {'messages': [AIMessage("Hello")]}
# Result: {'messages': [HumanMessage("Hi"), AIMessage("Hello")]}  # Accumulated!
```

### 5.3 Retriever Tool Implementation

**File:** `scripts/my_tools.py`

```python
from langchain_core.tools import tool
from scripts import utils
import os

@tool
def retrieve_docs(query: str, k: int = 5) -> str:
    """
    Retrieve relevant financial documents from ChromaDB.

    Extracts filters from query and retrieves matching documents.

    Args:
        query: Search query (e.g., "What was Amazon's revenue in Q2 2024?")
        k: Number of documents to retrieve (default: 5)

    Returns:
        Retrieved documents with metadata as formatted string
    """
    print(f"\n[TOOL] retrieve_docs called")
    print(f"[QUERY] {query}")

    # Extract filters and keywords
    filters = utils.extract_filters(query)
    ranking_keywords = utils.generate_ranking_keywords(query)

    # Fetch more docs for better re-ranking
    results = utils.search_docs(query, filters, ranking_keywords, k=10*k)

    # Re-rank with BM25Plus
    docs = utils.rank_documents_by_keywords(results, ranking_keywords, k=k)

    print(f"[RETRIEVED] {len(docs)} documents")

    # Handle empty results
    if len(docs) == 0:
        return f"No documents found for query: '{query}'. Try rephrasing."

    # Format documents
    retrieved_text = []
    for i, doc in enumerate(docs, 1):
        doc_text = [f"--- Document {i} ---"]

        # Add metadata
        for key, value in doc.metadata.items():
            doc_text.append(f"{key}: {value}")

        # Add content
        doc_text.append(f"\nContent:\n{doc.page_content}")

        text = "\n".join(doc_text)
        retrieved_text.append(text)

    retrieved_text = "\n\n".join(retrieved_text)

    # Save debug log
    os.makedirs("debug_logs", exist_ok=True)
    with open("debug_logs/retrieved_reranked_docs.md", "w", encoding='utf-8') as f:
        f.write(retrieved_text)

    return retrieved_text

# Test
result = retrieve_docs.invoke({"query": "Amazon revenue 2023", "k": 3})
print(result[:500])  # First 500 characters
```

**Output Format:**
```markdown
--- Document 1 ---
company_name: amazon
doc_type: 10-k
fiscal_year: 2023
page: 24
source_file: amazon 10-k 2023.pdf

Content:
## CONSOLIDATED STATEMENTS OF OPERATIONS
(In millions, except per share data)

| Year Ended December 31, | 2023 | 2022 | 2021 |
|-------------------------|------|------|------|
| Net revenue            | $574,785 | $513,983 | $469,822 |
...

--- Document 2 ---
...
```

### 5.4 Agent Node with System Prompt

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

LLM_MODEL = "qwen3"
BASE_URL = "http://localhost:11434"

llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL)

def agent_node(state: AgentState):
    """
    Agent node that processes messages and calls tools.
    """
    messages = state['messages']

    # Bind tools to LLM
    tools = [retrieve_docs]
    llm_with_tools = llm.bind_tools(tools)

    # System prompt (instruction manual for agent)
    system_prompt = """You are a financial document analysis assistant with access to document retrieval.

CRITICAL RULES:
1. ALWAYS call retrieve_docs tool FIRST - NEVER answer from memory
2. You MUST call the tool before providing financial information
3. Answer ONLY based on retrieved documents
4. If documents don't contain answer, clearly state that

WORKFLOW FOR SIMPLE QUESTIONS:
Step 1: Call retrieve_docs tool with user's question
Step 2: Wait for tool results
Step 3: Analyze retrieved documents
Step 4: Provide answer with citations (Company, Year, Quarter, Page)

WORKFLOW FOR COMPLEX/COMPARISON QUESTIONS:
Step 1: Break question into sub-questions
Example: "Compare Amazon and Google revenue" →
- Sub-question 1: "Amazon revenue"
- Sub-question 2: "Google revenue"

Step 2: Call retrieve_docs for EACH sub-question separately
- First call for Amazon
- Wait for results
- Second call for Google
- Wait for results

Step 3: Analyze all retrieved documents

Step 4: Present comparison in TABLE format:
| Metric | Company A | Company B |
|--------|-----------|-----------|
| Revenue | $X | $Y |

ANSWER FORMATTING (Use Markdown):
- Use **headings** (##, ###) for sections
- Use **paragraphs** for detailed findings
- Use **bullet points** for lists
- Use **tables** for comparisons and structured data
- Use **bold** for key metrics
- Cite sources: (Company: X, Year: Y, Quarter: Z, Page: N)

EXAMPLES:

Example 1 - Simple Question:
User: "What was Amazon's revenue in Q2 2024?"
You: [Call retrieve_docs] → [Analyze] →
"## Amazon Q2 2024 Revenue

Amazon's revenue for Q2 2024 was **$XXX billion**

**Source:** Amazon, 2024, Q2, Page 5"

Example 2 - Comparison Question:
User: "Compare Amazon and Google revenue"
You: [Call retrieve_docs("Amazon revenue")] → [Call retrieve_docs("Google revenue")] →
"## Revenue Comparison

| Company | Revenue | Year | Quarter |
|---------|---------|------|---------|
| Amazon  | $XXX B  | 2024 | Q2      |
| Google  | $YYY B  | 2024 | Q2      |

**Analysis:**
- Amazon's revenue was higher by $ZZZ billion

**Sources:**
- Amazon: 2024, Q2, Page 5
- Google: 2024, Q2, Page 8"

REMEMBER:
- ALWAYS call tool first
- Break complex questions into sub-questions
- Use tables for comparisons
- Format in detailed Markdown
- Always cite sources"""

    system_msg = SystemMessage(system_prompt)
    messages = [system_msg] + messages

    # Get LLM response
    response = llm_with_tools.invoke(messages)

    # Log tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            print(f"[AGENT] Called tool: {tc.get('name', '?')} with args: {tc.get('args', '?')}")
    else:
        print(f"[AGENT] Responding without tools")

    return {'messages': [response]}
```

### 5.5 Routing Logic

```python
def should_continue(state: AgentState):
    """
    Determine if agent should call tools or end.

    Returns:
        "tools" if agent wants to call tools
        END if agent is done
    """
    last_message = state['messages'][-1]

    # Check if last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return END
```

**Flow:**
```
Agent → has tool_calls? → YES → Tools Node → Agent
                       → NO  → END
```

### 5.6 Graph Creation

```python
def create_agent():
    """
    Create agentic RAG graph.
    """
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node('agent', agent_node)
    builder.add_node('tools', ToolNode([retrieve_docs]))

    # Add edges
    builder.add_edge(START, 'agent')
    builder.add_conditional_edges('agent', should_continue, ['tools', END])
    builder.add_edge('tools', 'agent')  # After tools, return to agent

    # Compile
    graph = builder.compile()

    return graph

agent = create_agent()
```

**Graph Visualization:**
```mermaid
START → agent → [conditional] → tools → agent
                             → END
```

### 5.7 Execution Examples

#### Example 1: Simple Query

```python
from langchain_core.messages import HumanMessage
from IPython.display import display, Markdown

query = "What is Amazon's revenue in 2023?"
result = agent.invoke({'messages': [HumanMessage(query)]})

# Display formatted output
display(Markdown(result['messages'][-1].content))
```

**Console Output:**
```
[AGENT] Called tool: retrieve_docs with args: {'query': "Amazon's revenue in 2023", 'k': 5}

[TOOL] retrieve_docs called
[QUERY] Amazon's revenue in 2023
   [1] Doc 17: score=23.4068
   [2] Doc 3: score=22.7389
   [3] Doc 8: score=20.1882
   [4] Doc 14: score=19.2649
   [5] Doc 5: score=18.0985
[RETRIEVED] 5 documents

[AGENT] Responding without tools
```

**Agent Response:**
```markdown
## Amazon's 2023 Revenue

Amazon's total revenue for 2023 was **$574.785 billion**, as reported in its consolidated net sales figures from the 10-K filing.

### Key Details:
- **Consolidated Net Sales**: $574,785 million
- **Year-over-Year Growth**: 12% increase compared to 2022
- **Segment Breakdown**:
  - North America: $352.828 billion (61% of total)
  - International: $131.200 billion (23% of total)
  - AWS: $90.757 billion (16% of total)

### Sources:
- **Document 2**: Amazon 10-K 2023, Page 24

This figure represents the sum of product and service sales across all segments.
```

#### Example 2: Comparison Query

```python
query = "Compare Amazon and Google revenue in 2023"
result = agent.invoke({'messages': [HumanMessage(query)]})

display(Markdown(result['messages'][-1].content))
```

**Console Output:**
```
[AGENT] Called tool: retrieve_docs with args: {'query': 'Amazon revenue 2023', 'k': 5}

[TOOL] retrieve_docs called
[QUERY] Amazon revenue 2023
[RETRIEVED] 5 documents

[AGENT] Called tool: retrieve_docs with args: {'query': 'Google revenue 2023', 'k': 5}

[TOOL] retrieve_docs called
[QUERY] Google revenue 2023
[RETRIEVED] 5 documents

[AGENT] Responding without tools
```

**Agent Response:**
```markdown
## 2023 Revenue Comparison: Amazon vs. Google

| Company | Total Revenue | YoY Growth | Primary Segments |
|---------|---------------|------------|------------------|
| **Amazon** | $574.785 billion | 12% | North America (61%), AWS (16%), International (23%) |
| **Google** | $307.394 billion | 9% | Google Services (89%), Google Cloud (11%) |

### Analysis:
- Amazon's revenue was **$267.4 billion higher** than Google's in 2023
- Amazon showed stronger growth (12% vs. 9%)
- Both companies demonstrated robust performance despite macroeconomic headwinds

### Revenue Breakdown:

**Amazon:**
- North America: $352.828 billion
- AWS: $90.757 billion
- International: $131.200 billion

**Google:**
- Google Services: $273.861 billion (Ads, YouTube, Play)
- Google Cloud: $33.533 billion

### Sources:
- Amazon: 2023, 10-K, Page 24
- Google: 2023, 10-K, Page 18
```

### 5.8 Multi-Turn Conversations

**Agentic RAG supports follow-up questions:**

```python
# Initial query
messages = [HumanMessage("What is Amazon's revenue in 2023?")]
result = agent.invoke({'messages': messages})

# Follow-up (appending to conversation)
messages = result['messages']  # Includes previous context
messages.append(HumanMessage("How does this compare to 2022?"))

result = agent.invoke({'messages': messages})
```

**Agent will:**
1. Recall previous context (2023 revenue)
2. Call `retrieve_docs("Amazon revenue 2022")`
3. Generate comparison

### 5.9 Advanced Prompt Engineering

**Handling Edge Cases:**

```python
system_prompt += """

EDGE CASES:

1. NO DOCUMENTS FOUND:
If tool returns "No documents found":
- Suggest rephrasing query
- Try without specific filters (year, quarter)
- Acknowledge limitation clearly

Example:
"I couldn't find documents matching your query about Tesla's 2024 Q3 revenue in our database. Our collection contains Amazon, Google, Apple, and Microsoft filings. Would you like information about one of these companies?"

2. AMBIGUOUS COMPANY NAMES:
- "GOOGL" → Google
- "MSFT" → Microsoft
- "AMZN" → Amazon

3. MISSING DATA POINTS:
If documents don't contain specific metric:
"The retrieved documents from Amazon's 2023 10-K (Pages 24-28) mention revenue figures but don't explicitly break down revenue by product category. The available data shows..."

4. CONFLICTING INFORMATION:
If multiple documents show different figures:
"Document 1 (Page 24) reports $574.785 billion while Document 2 (Page 51) shows $574,785 million (same figure in different units). These are consistent when converted."
"""
```

### 5.10 Production Optimizations

**Streaming Responses:**
```python
for chunk in agent.stream({'messages': [HumanMessage(query)]}):
    if 'agent' in chunk:
        message = chunk['agent']['messages'][-1]
        if hasattr(message, 'content'):
            print(message.content, end='', flush=True)
```

**Timeout Handling:**
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Agent execution timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

try:
    result = agent.invoke({'messages': [HumanMessage(query)]})
finally:
    signal.alarm(0)  # Cancel alarm
```

**Error Recovery:**
```python
def agent_node_with_retry(state: AgentState, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return agent_node(state)
        except Exception as e:
            print(f"[ERROR] Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return {'messages': [AIMessage(
                    f"I encountered an error: {e}. Please try rephrasing your question."
                )]}
```

### Key Takeaways
- Agentic RAG gives LLMs autonomy over retrieval decisions
- `@tool` decorator creates LangChain-compatible tools
- System prompts are instruction manuals for agents
- Agents can call tools multiple times for complex queries
- `Annotated[list, operator.add]` enables message accumulation
- Conditional routing creates agent ↔ tool loops
- Markdown formatting improves readability
- Multi-turn conversations maintain context

**Next:** Implement Corrective RAG (CRAG) with document grading and query rewriting

---

## Chapter 6: Corrective RAG (CRAG)

### Learning Objectives
- Implement document relevance grading before generation
- Rewrite queries for better retrieval
- Add web search fallback for missing information
- Prevent infinite retry loops (single rewrite)
- Use Pydantic models for grading decisions

### 6.1 CRAG Architecture

**Research Paper:** [Corrective Retrieval Augmented Generation (ArXiv 2401.15884)](https://arxiv.org/pdf/2401.15884.pdf)

**Problem with Basic RAG:**
```
Query → Retrieve → Generate
         ↑ (May retrieve irrelevant docs)
```

**CRAG Solution:**
```
Query → Retrieve → Grade Documents → Relevant? → Generate
                                   → Irrelevant? → Rewrite Query → Web Search → Generate
```

**Key Innovations:**
1. **Document Grading**: LLM evaluates retrieval quality
2. **Query Rewriting**: Improves search with better keywords
3. **Web Search Fallback**: Handles queries outside knowledge base
4. **Single Retry**: Prevents infinite loops

### 6.2 State Definition

```python
from typing_extensions import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    retrieved_docs: str          # Documents from retrieval
    is_relevant: bool            # Grading decision
    rewritten_query: str         # Improved query
```

**State Flow:**
```
messages: [HumanMessage("Query")]
         ↓
retrieved_docs: "Document 1...\nDocument 2..."
         ↓
is_relevant: True/False
         ↓ (if False)
rewritten_query: "Improved query with better keywords"
         ↓
retrieved_docs: "Web search results..."
         ↓
messages: [AIMessage("Answer")]
```

### 6.3 Retriever Node

```python
from scripts import my_tools

def retrieve_node(state: AgentState):
    """
    Retrieve documents using existing tools.
    """
    print(f"[RETRIEVE NODE] Fetching documents")

    user_question = state['messages'][-1].content
    print(f"[QUERY] {user_question}")

    # Call retrieve_docs tool
    result = my_tools.retrieve_docs.invoke({
        'query': user_question,
        'k': 5
    })

    print(f"[RETRIEVE NODE] Fetched relevant documents")

    # Save debug log
    with open('debug_logs/crag_retrieved_docs.md', 'w', encoding='utf-8') as f:
        f.write(f"Query: {user_question}\n\n")
        f.write(result)

    return {'retrieved_docs': result}
```

### 6.4 Document Grading

**Pydantic Schema:**
```python
from pydantic import BaseModel, Field

class GradeDecision(BaseModel):
    is_relevant: bool = Field(
        description="True if documents can answer question, False if irrelevant"
    )
    reasoning: str = Field(
        description="Brief explanation of grading decision"
    )
```

**Grading Node:**
```python
from langchain_ollama import ChatOllama

LLM_MODEL = "qwen3"
BASE_URL = "http://localhost:11434"
llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL)

def grade_node(state: AgentState):
    """
    Grade retrieved documents for relevance.

    Returns:
        {'is_relevant': True/False}
    """
    llm_structured = llm.with_structured_output(GradeDecision)

    user_question = state['messages'][-1].content
    retrieved_docs = state.get('retrieved_docs', '')

    prompt = f"""You are a document relevance grader.

TASK: Evaluate if retrieved documents can answer the user's question.

USER QUESTION: {user_question}

RETRIEVED DOCUMENTS:
{retrieved_docs}

GRADING CRITERIA:
- is_relevant = True: Documents contain information to answer question
- is_relevant = False: Documents are completely irrelevant or off-topic

Be permissive: If documents contain ANY relevant information, grade as relevant.

Output JSON:
{{
    "is_relevant": true/false,
    "reasoning": "Brief explanation..."
}}"""

    response = llm_structured.invoke(prompt)

    print(f"[GRADE] Relevant: {response.is_relevant}")
    print(f"[REASONING] {response.reasoning}")

    return {'is_relevant': response.is_relevant}
```

**Example Grading:**
```python
# Good retrieval
state = {
    'messages': [HumanMessage("Amazon's revenue in 2023?")],
    'retrieved_docs': "Document 1: Amazon's 2023 revenue was $574.785 billion..."
}

grade_node(state)
# Output:
# [GRADE] Relevant: True
# [REASONING] Documents explicitly state Amazon's 2023 revenue as $574.785 billion

# Poor retrieval
state = {
    'messages': [HumanMessage("Tesla's revenue in 2023?")],
    'retrieved_docs': "No documents found"
}

grade_node(state)
# Output:
# [GRADE] Relevant: False
# [REASONING] No documents retrieved to answer question about Tesla's revenue
```

### 6.5 Query Rewriting

```python
def rewrite_query_node(state: AgentState):
    """
    Rewrite query with better keywords for retry.
    """
    user_question = state['messages'][-1].content

    prompt = f"""You are a query rewriting expert.

TASK: Rewrite the user's question to be more specific and targeted.

ORIGINAL QUESTION: {user_question}

INSTRUCTIONS:
- Make query more specific with keywords
- Add relevant financial terms (revenue, profit, earnings, etc.)
- Include company names, years, quarters if mentioned
- Keep concise (one sentence)

Examples:
"Amazon earnings" → "What was Amazon's annual revenue in 2023 as reported in its 10-K filing?"
"Google profit" → "What was Google's net income for fiscal year 2023?"

Output ONLY the rewritten query, nothing else."""

    response = llm.invoke(prompt)
    rewritten_query = response.content.strip()

    print(f"[REWRITE] Original: {user_question}")
    print(f"[REWRITE] New: {rewritten_query}")

    return {'rewritten_query': rewritten_query}
```

**Example Rewriting:**
```
Original: "Amazon and goodles revenue in 2023?"
         (Typo: "goodles" instead of "Google")

Rewritten: "What was Amazon's and Google's annual revenue in 2023?"
```

### 6.6 Web Search Fallback

**File:** `scripts/my_tools.py`

```python
from duckduckgo_search import DDGS
from langchain_core.tools import tool

@tool
def web_search(query: str, num_results: int = 10) -> str:
    """
    Search web using DuckDuckGo.

    Args:
        query: Search query
        num_results: Number of results (default: 10)

    Returns:
        Formatted search results
    """
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=num_results))

        formatted_results = [f"Search results for query: '{query}'\n"]

        for i, result in enumerate(results, 1):
            formatted_results.append(f"\n{i}. **{result['title']}**")
            formatted_results.append(f"   {result['body']}")
            formatted_results.append(f"   {result['href']}")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Web search failed: {e}"

# Test
web_search.invoke({"query": "Amazon 2023 revenue", "num_results": 3})
```

**Output:**
```
Search results for query: 'Amazon 2023 revenue'

1. **Amazon.com Inc. Annual Revenue 2023 - Statista**
   Amazon's annual revenue in 2023 was $574.785 billion, a 11.83% increase from 2022.
   https://www.statista.com/statistics/266282/annual-net-revenue-of-amazoncom/

2. **Amazon Reports Fourth Quarter Results**
   Full year 2023 net sales increased 12% to $574.8 billion...
   https://ir.aboutamazon.com/news-release/

3. **Amazon 2023 Revenue Growth Analysis**
   Revenue grew 12% YoY driven by AWS, advertising, and third-party seller services.
   https://www.macrotrends.net/stocks/charts/AMZN/amazon/revenue
```

**Web Search Node:**
```python
def web_search_node(state: AgentState):
    """
    Search web with rewritten query.
    """
    user_question = state['messages'][-1].content
    rewritten_query = state.get("rewritten_query", user_question)

    print(f"[WEB SEARCH] Searching: {rewritten_query}")

    result = my_tools.web_search.invoke({'query': rewritten_query})

    # Save debug log
    with open('debug_logs/crag_retry_websearch_docs.md', 'w', encoding='utf-8') as f:
        f.write(f"Rewritten Query: {rewritten_query}\n\n")
        f.write(result)

    return {'retrieved_docs': result}
```

### 6.7 Answer Generation Node

```python
from langchain_core.messages import AIMessage

def answer_node(state: AgentState):
    """
    Generate final answer from retrieved documents.
    """
    user_question = state['messages'][-1].content
    retrieved_docs = state.get('retrieved_docs', '')

    prompt = f"""You are an expert financial analyst.

TASK: Provide a detailed answer using retrieved documents.

REQUIREMENTS:
1. Write 200-300 words
2. Use MARKDOWN formatting:
   - ## Headings
   - **Bold** for key metrics
   - Bullet points for lists
   - Tables for comparisons
3. Include inline citations [1], [2], [3]
4. Add References section: "Company: X, Year: Y, Quarter: Z, Page: N"
   OR if from web search: "[web_search]"

Be thorough and detailed.

USER QUESTION: {user_question}

RETRIEVED DOCUMENTS:
{retrieved_docs}

Provide detailed answer with citations:"""

    response = llm.invoke(prompt)

    return {'messages': [response]}
```

### 6.8 Router Logic

```python
def should_rewrite(state: AgentState):
    """
    Decide whether to rewrite query or proceed to answer.

    Returns:
        "answer" if documents are relevant
        "rewrite" if documents are irrelevant
    """
    is_relevant = state.get('is_relevant', True)

    if is_relevant:
        print(f"[ROUTER] Documents relevant → Proceeding to answer")
        return "answer"
    else:
        print(f"[ROUTER] Documents irrelevant → Rewriting query")
        return "rewrite"
```

### 6.9 Graph Construction

```python
from langgraph.graph import StateGraph, START, END

def create_crag_agent():
    """
    Create CRAG graph with grading and fallback.
    """
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node('retriever', retrieve_node)
    builder.add_node('grade', grade_node)
    builder.add_node('rewrite', rewrite_query_node)
    builder.add_node('web_search', web_search_node)
    builder.add_node('answer', answer_node)

    # Define edges
    builder.add_edge(START, 'retriever')
    builder.add_edge('retriever', 'grade')

    # Conditional: grade → answer OR rewrite
    builder.add_conditional_edges(
        'grade',
        should_rewrite,
        {
            'answer': 'answer',
            'rewrite': 'rewrite'
        }
    )

    # Rewrite path: rewrite → web_search → answer
    builder.add_edge('rewrite', 'web_search')
    builder.add_edge('web_search', 'answer')

    # End
    builder.add_edge('answer', END)

    graph = builder.compile()

    return graph

agent = create_crag_agent()
```

**Graph Flow:**
```mermaid
START → retriever → grade → [conditional]
                           ↓ (relevant)
                         answer → END
                           ↓ (irrelevant)
                        rewrite → web_search → answer → END
```

### 6.10 Execution Examples

#### Example 1: Successful Retrieval

```python
from langchain_core.messages import HumanMessage

query = "What is Amazon's revenue in 2023?"
result = agent.invoke({'messages': [HumanMessage(query)]})

result['messages'][-1].pretty_print()
```

**Console Output:**
```
[RETRIEVE NODE] Fetching documents
[QUERY] What is Amazon's revenue in 2023?
[TOOL] retrieve_docs called
[RETRIEVED] 5 documents

[GRADE] Relevant: True
[REASONING] Documents explicitly state Amazon's 2023 revenue as $574.785 billion

[ROUTER] Documents relevant → Proceeding to answer
```

**Agent Response:**
```markdown
## Amazon's Revenue in 2023

Amazon's total revenue for fiscal year 2023 was **$574.785 billion**, as reported in its 10-K filing. This represents a **12% year-over-year increase** compared to 2022.

### Breakdown by Segment
- **North America**: $352.828 billion (61%)
- **AWS**: $90.757 billion (16%)
- **International**: $131.200 billion (23%)

**References**
Company: Amazon, Year: 2023, Quarter: 4, Page: 24 [1]
```

#### Example 2: Failed Retrieval → Rewrite → Web Search

```python
query = "What is Tesla's revenue in 2023?"
result = agent.invoke({'messages': [HumanMessage(query)]})

result['messages'][-1].pretty_print()
```

**Console Output:**
```
[RETRIEVE NODE] Fetching documents
[QUERY] What is Tesla's revenue in 2023?
[TOOL] retrieve_docs called
[RETRIEVED] 0 documents

[GRADE] Relevant: False
[REASONING] No documents retrieved to answer question about Tesla

[ROUTER] Documents irrelevant → Rewriting query

[REWRITE] Original: What is Tesla's revenue in 2023?
[REWRITE] New: What is Tesla Inc.'s annual revenue for 2023 as reported in its 2023 annual financial report?

[WEB SEARCH] Searching: What is Tesla Inc.'s annual revenue for 2023...
```

**Agent Response:**
```markdown
## Tesla Revenue in 2023

As of the latest available data, **Tesla's 2023 annual revenue** is **$95.47 billion**, representing a **56% year-over-year (YoY) growth** [web_search]. This figure is derived from Tesla's official financial report released in Q1 2024.

### Key Revenue Breakdown
- **Vehicle Revenue**: $85.75 billion (~90% of total)
- **Energy Products Revenue**: $9.72 billion (solar + storage)

### Geographic Performance
- **North America**: $46.5 billion (49%)
- **Europe**: $32.3 billion (34%)
- **China**: $16.6 billion (17%)

**References**
[web_search] - Tesla Investor Relations, 2023 Annual Report
```

#### Example 3: Typo Correction

```python
query = "what is amazon's and goodles revenue in 2023?"
         # Typo: "goodles" instead of "Google"

result = agent.invoke({'messages': [HumanMessage(query)]})
```

**Console Output:**
```
[RETRIEVE NODE] Fetching documents
[QUERY] what is amazon's and goodles revenue in 2023?
[RETRIEVED] 0 documents

[GRADE] Relevant: False
[REASONING] No documents found for "goodles" (likely typo)

[REWRITE] Original: what is amazon's and goodles revenue in 2023?
[REWRITE] New: What was Amazon's and Google's annual revenue in 2023?

[WEB SEARCH] Searching: What was Amazon's and Google's annual revenue in 2023?
```

**Agent Response:**
```markdown
## Amazon's and Google's 2023 Revenue

### Amazon
Amazon's annual revenue in 2023 was **$574.785 billion**, an 11.83% increase from 2022 [web_search].

### Google (Alphabet)
Google's parent company Alphabet reported **$307.394 billion** in revenue for 2023, a 9% YoY increase [web_search].

### Comparison Table
| Company | 2023 Revenue | YoY Growth |
|---------|--------------|------------|
| Amazon  | $574.8B      | 12%        |
| Google  | $307.4B      | 9%         |

**References**
[web_search] - Amazon: Statista, Google: Alphabet Investor Relations
```

### 6.11 Why Only ONE Rewrite?

**Problem:** Infinite retry loops

```
Retrieve (fail) → Rewrite → Web Search → Retrieve (fail) → Rewrite → ...
```

**Solution:** Single rewrite + web search fallback

```python
# State tracks rewrite count implicitly
# After rewrite → web_search (no more retrieval)
# After web_search → answer (guaranteed end)

builder.add_edge('rewrite', 'web_search')  # No retrieval retry
builder.add_edge('web_search', 'answer')   # Guaranteed end
```

**Alternative (explicit counter):**
```python
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    retrieved_docs: str
    is_relevant: bool
    rewritten_query: str
    retry_count: int  # Track retries

def should_rewrite(state: AgentState):
    retry_count = state.get('retry_count', 0)

    if retry_count >= 1:
        print("[ROUTER] Max retries reached → Proceeding to answer")
        return "answer"

    is_relevant = state.get('is_relevant', True)

    if is_relevant:
        return "answer"
    else:
        return "rewrite"

def rewrite_query_node(state: AgentState):
    # ... rewrite logic ...
    return {
        'rewritten_query': rewritten_query,
        'retry_count': state.get('retry_count', 0) + 1
    }
```

### 6.12 Production Enhancements

**Multi-Source Grading:**
```python
class GradeDecision(BaseModel):
    is_relevant: bool
    confidence: float = Field(ge=0.0, le=1.0)  # 0-1 confidence score
    reasoning: str

def grade_node(state: AgentState):
    # ... grading logic ...

    # Only rewrite if low confidence
    if response.is_relevant and response.confidence < 0.7:
        print(f"[GRADE] Relevant but low confidence ({response.confidence:.2f}) → Augmenting with web search")
        # Augment vectorstore results with web search
```

**Hybrid Retrieval:**
```python
def hybrid_search_node(state: AgentState):
    """
    Combine vectorstore + web search results.
    """
    query = state['messages'][-1].content

    # Vectorstore retrieval
    vector_docs = my_tools.retrieve_docs.invoke({'query': query, 'k': 3})

    # Web search
    web_docs = my_tools.web_search.invoke({'query': query, 'num_results': 3})

    # Combine
    combined_docs = f"## Vector Store Results\n{vector_docs}\n\n## Web Search Results\n{web_docs}"

    return {'retrieved_docs': combined_docs}
```

### Key Takeaways
- CRAG adds quality control to RAG pipeline
- Document grading prevents hallucinations
- Query rewriting improves retrieval success
- Web search provides fallback for missing data
- Single retry prevents infinite loops
- Pydantic models ensure structured grading
- Debug logs aid troubleshooting

**Next:** Implement Reflexion for iterative self-refinement

---

## Chapter 7: MySQL Agent with LangGraph

### Learning Objectives
- Build SQL agents for natural language database queries
- Implement SQL query validation (prevent injection)
- Create automatic error correction with retry logic
- Design tools for schema retrieval, query generation, and execution
- Handle multi-table joins and complex queries

### 7.1 MySQL Agent Architecture

**Problem:** Users want to query structured data with natural language

**Solution:** LangGraph agent with SQL tools

**Workflow:**
```
User Question → Get Schema → Generate SQL → Validate → Execute
                                           ↓ (if error)
                                         Fix SQL → Validate → Execute
```

**Database:** `employees_db` (SQLite)
- 300,024 employees
- Tables: employees, departments, dept_emp, salaries, titles, dept_manager

### 7.2 Database Setup

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri('sqlite:///db/employees_db-full-1.0.6.db')

# Get table names
tables = db.get_usable_table_names()
print(tables)
# ['departments', 'dept_emp', 'dept_manager', 'employees', 'salaries', 'titles']

# Get full schema
SCHEMA = db.get_table_info()
print(SCHEMA)
```

**Schema Output:**
```sql
CREATE TABLE employees (
    emp_no INTEGER NOT NULL,
    birth_date DATE NOT NULL,
    first_name VARCHAR(14) NOT NULL,
    last_name VARCHAR(16) NOT NULL,
    gender TEXT NOT NULL,
    hire_date DATE NOT NULL,
    PRIMARY KEY (emp_no)
)

CREATE TABLE salaries (
    emp_no INTEGER NOT NULL,
    salary INTEGER NOT NULL,
    from_date DATE NOT NULL,
    to_date DATE NOT NULL,
    PRIMARY KEY (emp_no, from_date),
    FOREIGN KEY(emp_no) REFERENCES employees (emp_no)
)

-- ... other tables ...
```

### 7.3 SQL Tools Implementation

#### Tool 1: Get Database Schema

```python
from langchain_core.tools import tool

@tool
def get_database_schema(table_name: str = None):
    """
    Get database schema information for SQL query generation.
    Use this first to understand table structure.

    Args:
        table_name: Optional specific table name

    Returns:
        Schema information as string
    """
    if table_name:
        tables = db.get_usable_table_names()
        if table_name.lower() in [t.lower() for t in tables]:
            return db.get_table_info([table_name])
        else:
            return f"Error: Table '{table_name}' not found. Available: {', '.join(tables)}"
    else:
        return SCHEMA

# Usage
get_database_schema.invoke("employees")
```

#### Tool 2: Generate SQL Query

```python
from langchain_ollama import ChatOllama

LLM_MODEL = "gpt-oss"  # Model with reasoning capabilities
BASE_URL = "http://localhost:11434"
llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL, reasoning=True)

@tool
def generate_sql_query(question: str, schema_info: str = None):
    """
    Generate SQL SELECT query from natural language.
    Always use this after getting schema.

    Args:
        question: Natural language question
        schema_info: Schema context (optional, uses full schema if None)

    Returns:
        SQL query string
    """
    schema_to_use = schema_info if schema_info else SCHEMA

    prompt = f"""Based on this database schema:
{schema_to_use}

Generate a SQL query to answer this question: {question}

Rules:
- Use ONLY SELECT statements
- Include only existing columns and tables
- Add appropriate WHERE, GROUP BY, ORDER BY as needed
- Limit results to 10 rows unless specified
- Use proper SQLite syntax

Return ONLY the SQL query, nothing else."""

    response = llm.invoke(prompt)
    sql_query = response.content.strip()

    print(f"[TOOL] Generated SQL: {sql_query[:50]}...")
    return sql_query

# Example
generate_sql_query.invoke("How many employees are there?")
# Output: "SELECT COUNT(*) FROM employees;"
```

**GPT-OSS Reasoning:**
The gpt-oss model includes reasoning traces:

```python
response = llm.invoke("How many employees?")
print(response.content)  # SQL query
print(response.additional_kwargs['reasoning_content'])
# "User asks: 'how many employees are there?' We need to query employees table.
#  Use COUNT(*) to get total count. Query: SELECT COUNT(*) FROM employees;"
```

#### Tool 3: Validate SQL Query

```python
import re

@tool
def validate_sql_query(query: str):
    """
    Validate SQL query for safety and syntax.

    Returns:
        Cleaned query if valid, or error message
    """
    clean_query = query.strip()

    # Remove SQL code blocks
    clean_query = re.sub(r'```sql\s*', '', clean_query, flags=re.IGNORECASE)
    clean_query = re.sub(r'```\s*', '', clean_query)
    clean_query = clean_query.strip().rstrip(';')

    # Check 1: Only SELECT allowed
    if not clean_query.upper().startswith('SELECT'):
        return "Error: Only SELECT statements allowed"

    # Check 2: Block dangerous keywords
    dangerous_keywords = [
        'INSERT', 'UPDATE', 'DELETE', 'ALTER',
        'DROP', 'CREATE', 'TRUNCATE'
    ]

    for keyword in dangerous_keywords:
        if keyword in clean_query.upper():
            return f"Error: {keyword} operations not allowed"

    print("[TOOL] SQL query validated ✓")
    return clean_query

# Examples
validate_sql_query.invoke("SELECT * FROM employees LIMIT 5")
# Output: "SELECT * FROM employees LIMIT 5"

validate_sql_query.invoke("DROP TABLE employees")
# Output: "Error: DROP operations not allowed"
```

#### Tool 4: Execute SQL Query

```python
@tool
def execute_sql_query(sql_query: str):
    """
    Execute validated SQL query and return results.
    Only use after validation.

    Args:
        sql_query: SQL query string

    Returns:
        Query results or error message
    """
    # Validate first
    query = validate_sql_query.invoke(sql_query)

    if query.startswith('Error:'):
        return f"Query '{sql_query}' validation failed: {query}"

    try:
        result = db.run(query)

        if result:
            return f"Query Results: {result}"
        else:
            return "Query executed successfully but no results found"

    except Exception as e:
        return f"Execution error: {e}"

# Example
execute_sql_query.invoke("SELECT COUNT(*) FROM employees")
# Output: "Query Results: [(300024,)]"
```

#### Tool 5: Fix SQL Error

```python
@tool
def fix_sql_error(original_query: str, error_message: str, question: str):
    """
    Fix failed SQL query by analyzing error.
    Use when validation or execution fails.

    Args:
        original_query: Failed SQL query
        error_message: Error description
        question: Original natural language question

    Returns:
        Corrected SQL query
    """
    fix_prompt = f"""The following SQL query failed:
Query: {original_query}
Error: {error_message}
Original Question: {question}

Database Schema:
{SCHEMA}

Analyze the error and provide corrected SQL that:
1. Fixes the specific error
2. Still answers original question
3. Uses only valid tables and columns from schema
4. Follows SQLite syntax

Return ONLY the corrected SQL query."""

    response = llm.invoke(fix_prompt)
    query = response.content.strip()

    print(f"[TOOL] Generated fixed SQL")
    return query

# Example
fix_sql_error.invoke({
    "original_query": "SELECT AVG(salary) FROM salaries WHERE to_date = 'current'",
    "error_message": "Invalid date format 'current'",
    "question": "What's the average current salary?"
})
# Output: "SELECT AVG(salary) FROM salaries WHERE to_date = '9999-01-01'"
```

### 7.4 Agent State and Graph

```python
from typing_extensions import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

tools = [
    get_database_schema,
    generate_sql_query,
    execute_sql_query,
    fix_sql_error
]

llm_with_tools = llm.bind_tools(tools)
```

### 7.5 Agent Node with System Prompt

```python
from langchain_core.messages import SystemMessage

def agent_node(state: AgentState):
    """
    SQL agent that uses tools to query database.
    """
    system_prompt = f"""You are an expert SQL analyst working with employees database.

Database Schema:
{SCHEMA}

Your workflow:
1. Use `get_database_schema` to understand tables/columns (if needed)
2. Use `generate_sql_query` to create SQL from question
3. Use `execute_sql_query` to run validated query
4. If error, use `fix_sql_error` to correct and retry (up to 3 times)

Rules:
- Follow workflow step by step
- If query fails, use fix tool and retry
- Provide clear, informative answers
- Be precise with table/column names
- Handle errors gracefully
- If fail after 3 attempts, explain what went wrong

Available tools:
- get_database_schema: Get table structure
- generate_sql_query: Create SQL from question
- execute_sql_query: Run query
- fix_sql_error: Fix failed queries

Remember: Always validate queries before execution."""

    messages = [SystemMessage(system_prompt)] + state['messages']
    response = llm_with_tools.invoke(messages)

    return {'messages': [response]}
```

### 7.6 Router and Graph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

def should_continue(state: AgentState):
    """
    Route to tools or end.
    """
    last_message = state['messages'][-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("[TOOL] Calling tools")
        for tc in last_message.tool_calls:
            print(f"  - {tc['name']} with args: {tc['args']}")
        return 'tools'
    else:
        print("[AGENT] Processing response")
        return END

def create_sql_agent():
    """
    Create SQL agent graph.
    """
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node('agent', agent_node)
    builder.add_node('tools', ToolNode(tools))

    # Add edges
    builder.add_edge(START, 'agent')
    builder.add_edge('tools', 'agent')
    builder.add_conditional_edges('agent', should_continue, ['tools', END])

    graph = builder.compile()
    return graph

agent = create_sql_agent()
```

### 7.7 Execution Examples

#### Example 1: Simple Count

```python
from langchain_core.messages import HumanMessage

query = "How many employees are there?"
result = agent.invoke({'messages': [HumanMessage(query)]})

result['messages'][-1].pretty_print()
```

**Console Output:**
```
[TOOL] Calling tools
  - get_database_schema with args: {'table_name': 'employees'}

[TOOL] Calling tools
  - generate_sql_query with args: {'question': 'how many employees are there', 'schema_info': 'CREATE TABLE employees...'}
[TOOL] Generated SQL: SELECT COUNT(*) AS employee_count FROM employees

[TOOL] Calling tools
  - execute_sql_query with args: {'sql_query': 'SELECT COUNT(*) AS employee_count FROM employees;'}
[TOOL] SQL query validated ✓

[AGENT] Processing response
```

**Agent Response:**
```
There are **300,024** employees in the database.
```

#### Example 2: Average Salary by Department

```python
query = "What is the average salary of each department? Show me top 5."
result = agent.invoke({'messages': [HumanMessage(query)]})

result['messages'][-1].pretty_print()
```

**Generated SQL:**
```sql
SELECT
    d.dept_name,
    AVG(s.salary) AS avg_salary
FROM departments d
JOIN dept_emp de ON d.dept_no = de.dept_no
JOIN salaries s ON de.emp_no = s.emp_no
WHERE de.to_date = '9999-01-01'  -- Current assignments
  AND s.to_date = '9999-01-01'   -- Current salaries
GROUP BY d.dept_no, d.dept_name
ORDER BY avg_salary DESC
LIMIT 5;
```

**Agent Response:**
```markdown
Here are the five departments with highest average current salary:

| Rank | Department  | Average Salary |
|------|-------------|----------------|
| 1    | Sales       | $88,852.97     |
| 2    | Marketing   | $80,058.85     |
| 3    | Finance     | $78,559.94     |
| 4    | Research    | $67,913.37     |
| 5    | Production  | $67,843.30     |

*Figures based on current employee assignments and salary records (to_date = '9999-01-01').*
```

#### Example 3: Complex Query with Joins

```python
query = "Show me the top 5 highest paid employees with their title and salaries"
result = agent.invoke({'messages': [HumanMessage(query)]})
```

**Generated SQL:**
```sql
SELECT
    e.emp_no,
    e.first_name,
    e.last_name,
    t.title,
    s.max_salary AS salary
FROM employees AS e
JOIN (
    SELECT emp_no, MAX(salary) AS max_salary
    FROM salaries
    GROUP BY emp_no
) AS s ON e.emp_no = s.emp_no
JOIN titles AS t ON e.emp_no = t.emp_no
WHERE t.to_date = '9999-01-01'  -- Current title
ORDER BY s.max_salary DESC
LIMIT 5;
```

**Agent Response:**
```markdown
**Top 5 Highest-Paid Employees:**

| Rank | Emp No | Full Name          | Title         | Salary  |
|------|--------|--------------------|---------------|---------|
| 1    | 43624  | Tokuyasu Pesch     | Senior Staff  | $158,220|
| 2    | 254466 | Honesty Mukaidono  | Senior Staff  | $156,286|
| 3    | 47978  | Xiahua Whitcomb    | Senior Staff  | $155,709|
| 4    | 253939 | Sanjai Luders      | Senior Staff  | $155,513|
| 5    | 109334 | Tsutomu Alameldin  | Senior Staff  | $155,377|

*Based on current salary periods (to_date = '9999-01-01').*
```

#### Example 4: Error Correction

```python
query = "Average salary by department show me top 5"
# Agent generates query with error, then fixes it
```

**First Attempt (Error):**
```sql
SELECT
    d.dept_name,
    AVG(s.salary) AS avg_salary
FROM departments d
JOIN dept_emp de ON d.dept_no = de.dept_no
JOIN salaries s ON de.emp_no = s.emp_no
WHERE de.current = true  -- ERROR: No 'current' column
GROUP BY d.dept_name
ORDER BY avg_salary DESC
LIMIT 5;
```

**Error:** `no such column: de.current`

**Fix Tool Called:**
```
[TOOL] Calling tools
  - fix_sql_error with args: {
      'original_query': '...',
      'error_message': 'no such column: de.current',
      'question': 'Average salary by department top 5'
    }
[TOOL] Generated fixed SQL
```

**Corrected SQL:**
```sql
SELECT
    d.dept_name,
    AVG(s.salary) AS avg_salary
FROM departments d
JOIN dept_emp de ON d.dept_no = de.dept_no
JOIN salaries s ON de.emp_no = s.emp_no
WHERE de.to_date = '9999-01-01'  -- FIXED: Use to_date for current
  AND s.to_date = '9999-01-01'
GROUP BY d.dept_name
ORDER BY avg_salary DESC
LIMIT 5;
```

**Second Attempt:** ✓ Success

### 7.8 Production Enhancements

**Retry Limiting:**
```python
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    retry_count: int  # Track SQL fix attempts

def agent_node(state: AgentState):
    retry_count = state.get('retry_count', 0)

    if retry_count >= 3:
        return {'messages': [AIMessage(
            "I've attempted to fix the SQL query 3 times but still encountering errors. "
            "Please rephrase your question or provide more specific details."
        )]}

    # ... normal agent logic ...
```

**Query Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def execute_sql_cached(query: str):
    return db.run(query)
```

**Query Explanation:**
```python
@tool
def explain_sql_query(sql_query: str):
    """
    Explain what SQL query does in plain English.
    """
    prompt = f"""Explain this SQL query in simple terms:

{sql_query}

Provide:
1. What data it retrieves
2. Which tables it uses
3. Any filters applied
4. How results are sorted"""

    response = llm.invoke(prompt)
    return response.content
```

### Key Takeaways
- SQL agents enable natural language database queries
- Schema retrieval helps LLM understand data structure
- Query validation prevents SQL injection attacks
- Automatic error correction with retry logic improves success rate
- GPT-OSS model provides reasoning traces for debugging
- Multi-table joins require careful schema understanding
- Current records use `to_date = '9999-01-01'` convention

**Database:** 300,024 employees across 6 tables with salaries, departments, titles

---

# Course Summary and Key Technical Terms

## Technologies Mastered

### LangGraph Orchestration
- **StateGraph**: Workflow builder with typed states
- **TypedDict**: Type-safe state schemas
- **Annotated[list, operator.add]**: Message accumulation pattern
- **Conditional Edges**: Dynamic routing based on runtime decisions
- **ToolNode**: Automated tool execution nodes
- **START/END**: Graph entry/exit points

### Ollama Local LLMs
- **Models**: Qwen3, GPT-OSS (reasoning), Llama3.2, Qwen3:0.6b
- **Embeddings**: nomic-embed-text (384 dimensions)
- **Modelfiles**: Custom model creation with system prompts
- **API Endpoints**: /api/generate, /api/chat, /api/embed
- **Privacy**: On-premise inference, zero external API calls

### Vector Storage
- **ChromaDB**: Persistent vector database with collections
- **Metadata Filtering**: AND/OR logic with complex filters
- **MMR Retrieval**: Maximal Marginal Relevance for diversity
- **Collection**: financial_docs with 1,270+ pages

### Document Processing
- **Docling**: PDF to markdown with OCR, GPU acceleration
- **Page-wise Ingestion**: Granular retrieval and precise citations
- **SHA-256 Hashing**: File deduplication
- **Metadata Extraction**: Company, document type, fiscal year/quarter

### Re-Ranking
- **BM25Plus**: Probabilistic ranking algorithm
- **Heading Extraction**: Markdown structure analysis
- **Term Frequency**: IDF weighting for keyword importance
- **Two-Stage Retrieval**: Vector search + BM25 re-ranking

### Structured Outputs
- **Pydantic BaseModel**: Type-safe data validation
- **Field**: Constraints (ge, le, min_length, max_length)
- **Enum**: Strict categorical values
- **with_structured_output()**: Force LLM to return Pydantic models
- **model_dump(exclude_none=True)**: Clean serialization

### RAG Architectures
- **PageRAG**: Page-wise document processing with metadata
- **Agentic RAG**: Tool-driven retrieval with multi-turn reasoning
- **Corrective RAG (CRAG)**: Document grading + query rewriting + web fallback
- **Reflexion**: Iterative self-refinement with completeness checks
- **Self-RAG**: Multi-point quality validation (relevance, grounding, answer quality)
- **Adaptive RAG**: Multi-datasource routing (vectorstore, SQL, web)

### SQL Integration
- **SQLDatabase**: LangChain abstraction for databases
- **Natural Language to SQL**: LLM-powered query generation
- **Query Validation**: SQL injection prevention
- **Error Correction**: Automatic retry with fix_sql_error
- **Multi-Table Joins**: Complex queries across 6 tables

### Tools and Agents
- **@tool Decorator**: Create LangChain-compatible tools
- **bind_tools()**: Attach tools to LLMs
- **ToolNode**: Automatic tool execution in graphs
- **ReAct Pattern**: Reasoning + Acting agents

## Key Metrics
- **Dataset**: 1,270+ financial document pages (Amazon, Google, Apple, Microsoft)
- **Database**: 300,024 employees across 6 tables
- **Embedding Model**: 384-dimensional vectors (nomic-embed-text)
- **Retrieval**: k=3-5 final docs from fetch_k=60-100 candidates
- **Re-Ranking**: BM25Plus on heading+content chunks

## Production Patterns
- **Debug Logging**: Markdown files in debug_logs/
- **Caching**: LRU cache for LLM calls and database queries
- **Error Handling**: Try/except with retry logic
- **Monitoring**: Timing metrics for each pipeline stage
- **Streaming**: Real-time response generation
- **State Persistence**: MemorySaver for multi-session conversations

## Course Outcomes
Students will be able to:
1. Deploy private LLMs with Ollama for production RAG systems
2. Build multi-step agent workflows with LangGraph state machines
3. Implement advanced RAG patterns (CRAG, Reflexion, Self-RAG, Adaptive)
4. Process financial documents with Docling and ChromaDB
5. Re-rank results with BM25Plus for improved precision
6. Create SQL agents for natural language database queries
7. Design effective system prompts for financial analysis
8. Handle multi-turn conversations with state management
9. Implement fallback mechanisms (web search, query rewriting)
10. Build production-ready systems with error handling and monitoring

---

**Total Course Duration:** 10-12 hours (advanced level)
**Projects:** 7 major RAG architectures + 1 SQL agent
**Code Files:** 12+ Jupyter notebooks with complete implementations
**Real-World Dataset:** SEC filings (10-K, 10-Q, 8-K) from major tech companies

---

*This comprehensive course documentation covers all technical aspects taught in the Private Agentic RAG with LangGraph and Ollama course, focusing on advanced RAG applications, MySQL agent integration, Ollama setup, and LangGraph fundamentals for production-ready AI systems.*