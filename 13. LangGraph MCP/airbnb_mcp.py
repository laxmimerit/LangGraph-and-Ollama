"""Simple Airbnb MCP module."""
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage

from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

from typing_extensions import TypedDict, Annotated
import operator

# MCP GITHUB
# https://github.com/laxmimerit/MCP-Mastery-with-Claude-and-Langchain
# https://github.com/langchain-ai/langchain-mcp-adapters

# Config
LLM_MODEL = "qwen3"
BASE_URL = "http://localhost:11434"

llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL, temperature=0)

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

async def create_agent():

    client = MultiServerMCPClient(
    {
        "airbnb": {
            "command": "npx",
            "args": [
                "-y",
                "@openbnb/mcp-server-airbnb",
                "--ignore-robots-txt"
            ],
            "transport": "stdio"
    }
    }
)
    
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} Tools.: {tools}")

    
    # create in-function agent node
    def agent(state: AgentState):
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(state['messages'])
        return {'messages': [response]}

    builder = StateGraph(AgentState)
    builder.add_node('agent', agent)
    builder.add_node('tools', ToolNode(tools))

    # add edges
    builder.add_edge(START, 'agent')
    builder.add_edge('tools', 'agent')
    builder.add_conditional_edges('agent', tools_condition)

    graph = builder.compile()
    return graph

async def search(query):
    agent = await create_agent()
    result = await agent.ainvoke({'messages': [HumanMessage(query)]})
    response = result['messages'][-1].content
    print(f"\n{response}\n")

    return response

if __name__=="__main__":
    # asyncio.run(create_agent())
    asyncio.run(search("Find premium hotels in Mumbai"))