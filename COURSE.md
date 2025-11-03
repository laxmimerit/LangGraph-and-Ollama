# LangGraph and Ollama Course

A hands-on course for building AI agents and RAG systems with LangChain, LangGraph, and Ollama.

---

## Lesson 02: LangChain Getting Started

**Topics:**
- LangChain installation and setup
- Environment configuration with LangSmith
- Working with Ollama chat models
- Basic LLM invocation and streaming

**Key Concepts:** LangChain fundamentals, Ollama integration, environment setup

---

## Lesson 03: LangGraph Getting Started

**Topics:**
- Introduction to LangGraph concepts (State, Nodes, Edges)
- Creating stateful workflows with StateGraph
- Building linear processing pipelines
- Compiling and running graphs

**Key Concepts:** State management, graph building, workflow orchestration

---

## Lesson 04: Conditional Routing

**Topics:**
- LLM integration in LangGraph workflows
- Pydantic models for structured LLM outputs
- Conditional routing based on LLM decisions
- Sentiment analysis with dynamic responses

**Key Concepts:** `with_structured_output()`, conditional edges, sentiment routing

**Example:** Twitter customer support agent with sentiment-based routing

---

## Lesson 05: LangGraph Agent with Tools

**Topics:**
- Creating tools with `@tool` decorator
- ReAct pattern (Reasoning + Acting)
- Tool binding with `bind_tools()`
- ToolNode for automatic tool execution

**Key Concepts:** Function calling, tool integration, agent decision-making

**Example:** Agent with weather, calculator, and documentation search tools

---

## Lesson 06: Agentic Memory and Streaming

**Topics:**
- Conversation memory with checkpointers
- Thread management for multi-session conversations
- Streaming responses for real-time UX
- MemorySaver for persistent state

**Key Concepts:** Checkpointers, thread_id, streaming, stateful conversations

**Example:** Multi-session chatbot with memory and tool access

---

## Lesson 07: PageRAG - Data Ingestion

**Topics:**
- PDF page extraction with Docling
- LLM-based metadata extraction
- ChromaDB vector store setup
- Document deduplication with file hashing
- Structured metadata schemas with Pydantic

**Key Concepts:** Page-wise processing, structured metadata, vector databases, RAG pipeline

**Example:** Financial document ingestion (10-K, 10-Q filings) with rich metadata

---

## Lesson 08: Enhanced Agentic PageRAG

**Topics:**
- Building retriever tools for agents
- Query metadata extraction with structured output
- Filter-based retrieval with ChromaDB
- ToolNode pattern for visible tool execution
- Dynamic metadata display

**Key Concepts:** Agentic RAG, query analysis, metadata filtering, tool-based retrieval

**Example:** Financial analysis agent with intelligent document retrieval and comparison capabilities

---

## Prerequisites

- Python 3.8+
- Ollama installed and running
- Basic Python knowledge
- Understanding of LLMs

## Models Used

- **LLM:** Qwen3, Qwen3:0.6b
- **Embeddings:** nomic-embed-text
