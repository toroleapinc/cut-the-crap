"""
Session 5 Hands-On: Multi-Step Research Agent with LangGraph
Cut the Crap — AI Engineer Edition

Updated: February 2026
Uses: LangGraph (latest), GPT-4.1

This agent:
1. Takes a research topic
2. Searches for information
3. Analyzes findings and decides if more research is needed
4. Writes a final summary

Requirements:
    pip install langgraph langchain-openai langchain-core
    export OPENAI_API_KEY=your-key
"""

import json
import os
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    research_count: int  # Track how many searches we've done


# --- Tools ---
@tool
def search_web(query: str) -> str:
    """Search the web for information on a topic. Returns relevant results."""
    # In production, use Tavily, Brave, SerpAPI, etc.
    # This is a simulation for the hands-on exercise.
    fake_results = {
        "ai regulations": (
            "The EU AI Act came into force in August 2024. It classifies AI systems by risk level. "
            "High-risk systems (hiring, credit scoring, law enforcement) face strict requirements. "
            "The US has executive orders on AI safety. China requires algorithm registration. "
            "As of 2026, enforcement of the EU AI Act is ramping up with fines for non-compliance."
        ),
        "ai agents": (
            "AI agents are LLMs that can use tools and make decisions in a loop. "
            "Popular frameworks include LangGraph, CrewAI, OpenAI Agents SDK, and Anthropic's tool use. "
            "Key challenges: reliability, cost control, and safety guardrails. "
            "In 2025-2026, agentic coding (Codex, Claude Code) became mainstream."
        ),
        "default": (
            f"Search results for '{query}': Multiple sources discuss this topic. "
            "Key findings include recent developments, industry trends, and expert opinions. "
            "Further research may be needed for specific aspects."
        ),
    }
    query_lower = query.lower()
    for key, result in fake_results.items():
        if key in query_lower:
            return result
    return fake_results["default"]


@tool
def write_report(title: str, content: str) -> str:
    """Write a final research report. Use this when you have enough information."""
    report = f"\n{'='*60}\n📄 RESEARCH REPORT: {title}\n{'='*60}\n\n{content}\n\n{'='*60}"
    return report


# --- LLM Setup ---
# GPT-4.1 is the best non-reasoning model as of Feb 2026
llm = ChatOpenAI(model="gpt-4.1", temperature=0)
llm_with_tools = llm.bind_tools([search_web, write_report])


# --- Graph Nodes ---
SYSTEM_PROMPT = """You are a thorough research assistant. Your process:
1. Search for information on the topic (use search_web tool)
2. Analyze what you've found
3. If you need more specific information, search again (but max 3 searches)
4. When you have enough info, write a report using the write_report tool
5. After writing the report, give a brief verbal summary

Be thorough but efficient. Don't repeat searches."""


def agent_node(state: AgentState) -> dict:
    """The main agent reasoning node."""
    messages = state["messages"]

    # Add system prompt if not present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route: if the LLM wants to use tools, go to tools. Otherwise, end."""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def count_research(state: AgentState) -> dict:
    """Track how many times we've searched (prevents infinite loops)."""
    count = state.get("research_count", 0)
    # Count search_web calls in the latest tool calls
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls"):
        for tc in last_msg.tool_calls:
            if tc["name"] == "search_web":
                count += 1
    return {"research_count": count}


# --- Build the Graph ---
def build_agent():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode([search_web, write_report]))
    graph.add_node("counter", count_research)

    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, ["tools", END])
    graph.add_edge("tools", "counter")
    graph.add_edge("counter", "agent")

    return graph.compile()


# --- Run It ---
def main():
    agent = build_agent()

    # Visualize the graph (if you have graphviz)
    try:
        print(agent.get_graph().draw_ascii())
    except Exception:
        pass

    print("\n🔬 Research Agent Ready!")
    print("=" * 40)

    topic = input("\nEnter a research topic (or press Enter for default): ").strip()
    if not topic:
        topic = "What are the current AI regulations worldwide?"

    print(f"\n🔍 Researching: {topic}\n")

    # Run the agent
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=topic)],
            "research_count": 0,
        }
    )

    # Print the final response
    print("\n" + "=" * 60)
    print("💬 Agent's Final Response:")
    print("=" * 60)
    final_message = result["messages"][-1]
    print(final_message.content)

    # Show stats
    print(f"\n📊 Total messages exchanged: {len(result['messages'])}")
    print(f"🔍 Research iterations: {result.get('research_count', 0)}")

    # Show the full conversation for learning purposes
    show_trace = input("\nShow full message trace? (y/n): ").strip().lower()
    if show_trace == "y":
        print("\n" + "=" * 60)
        print("📝 Full Message Trace:")
        print("=" * 60)
        for i, msg in enumerate(result["messages"]):
            role = msg.__class__.__name__.replace("Message", "")
            content = str(msg.content)[:200]
            tool_calls = ""
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls = f" [Tools: {', '.join(tc['name'] for tc in msg.tool_calls)}]"
            print(f"\n  [{i}] {role}{tool_calls}")
            print(f"      {content}")


if __name__ == "__main__":
    main()
