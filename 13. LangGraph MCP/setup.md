# Airbnb MCP Server Setup

Quick setup guide for using Airbnb MCP server with LangGraph.

## Prerequisites
```bash
# 1. Install Node.js (required for npx)
# Download from: https://nodejs.org/

# 2. Install Python dependencies
pip install langchain-mcp-adapters langgraph langchain-ollama

# 3. Ensure Ollama is running
ollama pull qwen3
```

## Installation

The Airbnb MCP server is automatically installed via npx when first run. No manual installation needed.

## Configuration

### Basic Setup (Default)
```python
client = MultiServerMCPClient({
    "airbnb": {
        "command": "npx",
        "args": ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
        "transport": "stdio",
    }
})
```

## Available Tools

### 1. `airbnb_search`
Search Airbnb listings with filters.

**Parameters:**
- `location` (required): "San Francisco, CA"
- `checkin`: "2025-12-01"
- `checkout`: "2025-12-05"
- `adults`: 2
- `children`: 0
- `pets`: 1
- `minPrice`: 50
- `maxPrice`: 200

### 2. `airbnb_listing_details`
Get detailed property information.

**Parameters:**
- `id` (required): Airbnb listing ID
- `checkin`, `checkout`, `adults`, etc. (optional)

## Usage Example
```python
query = "Find pet-friendly Airbnb in NYC under $200/night"
result = await graph.ainvoke({"messages": [HumanMessage(content=query)]})
```

## Run
```bash
python airbnb_agent.py
```

## Notes

- First run downloads @openbnb/mcp-server-airbnb via npx
- Respects robots.txt by default (use `--ignore-robots-txt` for testing)
- Rate limiting: Be respectful with request frequency
- Not affiliated with Airbnb, Inc.

## Troubleshooting

**npx not found:**
```bash
# Install Node.js from nodejs.org
node --version  # Should show v18+
```

**Connection timeout:**
- Check internet connection
- Airbnb may be rate-limiting requests

**Tool not found:**
- Ensure npx installed @openbnb/mcp-server-airbnb successfully
- Check terminal for error messages