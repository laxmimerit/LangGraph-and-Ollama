"""Simple Airbnb MCP module."""
import asyncio
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage

# Config
LLM_MODEL = "qwen3"
BASE_URL = "http://localhost:11434"

async def create_graph():
    """Create Airbnb graph."""
    client = MultiServerMCPClient({
        "airbnb": {
            "command": "npx",
            "args": ["-y", "@openbnb/mcp-server-airbnb"],
            "transport": "stdio",
        }
    })
    
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tools")
    
    llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL, temperature=0)
    
    def agent(state):
        return {"messages": [llm.bind_tools(tools).invoke(state["messages"])]}
    
    builder = StateGraph(dict)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    
    return builder.compile()

async def search(query):
    """Search Airbnb."""
    graph = await create_graph()
    result = await graph.ainvoke({"messages": [HumanMessage(content=query)]})
    response = result["messages"][-1].content
    print(f"\n{response}\n")
    return response

if __name__ == "__main__":
    asyncio.run(search("Find Airbnb in San Francisco"))