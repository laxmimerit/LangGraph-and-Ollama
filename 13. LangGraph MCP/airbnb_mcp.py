"""Simple Airbnb MCP module."""

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage

from typing_extensions import TypedDict, Annotated
import operator

# Config
LLM_MODEL = "qwen3"
BASE_URL = "http://localhost:11434"

llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL, temperature=0)


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


######## MCP SETUP ###############
# MCP GITHUB
# https://github.com/laxmimerit/MCP-Mastery-with-Claude-and-Langchain
# https://github.com/langchain-ai/langchain-mcp-adapters

from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio


async def get_tools():

    client = MultiServerMCPClient(
        {
            "airbnb": {
                "command": "npx",
                "args": ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
                "transport": "stdio"
            }
        }
    )


    tools = await client.get_tools()

    print(f"Loaded {len(tools)} Tools")
    print(f"Tools Available: {tools}")

    return tools


async def agent_node(state: AgentState):
    tools = await get_tools()

    llm_with_tools = llm.bind_tools(tools)

    response = llm_with_tools.invoke(state['messages'])

    return {'messages': [response]}


async def create_agent():
    tools = await get_tools()

    builder = StateGraph(AgentState)
    builder.add_node('agent', agent_node)
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

    print(response)

    return response


if __name__=="__main__":
    query = "Show me premium hotels for party in Mumbai"
    asyncio.run(search(query))