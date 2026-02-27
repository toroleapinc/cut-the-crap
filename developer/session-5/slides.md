# Session 5: Agentic AI & Frameworks
## Cut the Crap — AI Engineer Edition

---

## Slide 1: What We're Covering Today

- What makes something an "agent" vs just a chatbot
- The ReAct loop — how agents actually think
- Framework landscape: LangGraph, CrewAI, OpenAI SDK, and more
- Multi-agent orchestration
- Agent memory
- **Hands-on:** Build a multi-step agent with LangGraph

> **SPEAKER NOTES:**
> "Last session we connected tools via MCP. Today we go further — we're building things that can *decide* which tools to use, in what order, and loop until they solve a problem. That's what an agent is. Fair warning: the word 'agent' is the most overhyped term in AI right now, so let's cut through the noise."

---

## Slide 2: What Is an Agent?

**An agent is an LLM that can:**
1. Observe — take in information
2. Think — reason about what to do
3. Act — use tools or produce output
4. Repeat — loop until the task is done

**What an agent is NOT:**
- A chatbot with a system prompt ❌
- An API call with JSON output ❌
- A fixed pipeline with no decisions ❌

> **SPEAKER NOTES:**
> "If your code calls an LLM once and returns the result, that's not an agent. If it calls an LLM, the LLM decides to search the web, reads the results, decides it needs more info, searches again, then writes a summary — THAT's an agent. The key differentiator is the loop and the autonomy to decide next steps."

---

## Slide 3: The ReAct Loop

```
┌─────────────────────────────────────┐
│          User gives a task          │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  REASON: LLM thinks about next step │◄──┐
└──────────────┬───────────────────────┘   │
               │                           │
               ▼                           │
┌──────────────────────────────────────┐   │
│  ACT: Execute a tool / take action   │   │
└──────────────┬───────────────────────┘   │
               │                           │
               ▼                           │
┌──────────────────────────────────────┐   │
│  OBSERVE: Read the result            │───┘
└──────────────┬───────────────────────┘
               │ (done?)
               ▼
┌──────────────────────────────────────┐
│         Final answer to user         │
└──────────────────────────────────────┘
```

> **SPEAKER NOTES:**
> "ReAct stands for Reason + Act. It's from a 2022 paper but the pattern is everywhere now. Every agent framework implements some version of this. The LLM reasons about what to do, takes an action (usually a tool call), observes the result, and decides whether to keep going or return a final answer. This is the fundamental pattern — everything else is just scaffolding around it."

---

## Slide 4: ReAct in Raw Code (No Framework)

```python
import openai

client = openai.OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]

messages = [
    {"role": "system", "content": "You are a research assistant."},
    {"role": "user", "content": "What's the population of Tokyo vs NYC?"}
]

# The ReAct loop — it's just a while loop!
while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools
    )
    
    msg = response.choices[0].message
    messages.append(msg)
    
    if msg.tool_calls:
        for call in msg.tool_calls:
            result = execute_tool(call.function.name, call.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result
            })
    else:
        print(msg.content)
        break  # No more tool calls = we're done
```

> **SPEAKER NOTES:**
> "Before we look at any framework, I want you to see that an agent is literally just a while loop. The LLM decides whether to call a tool or give a final answer. If it calls a tool, we execute it, feed the result back, and let the LLM decide again. That's it. Every framework is just a fancier version of this loop with error handling, state management, and routing. You could build agents with zero frameworks."

---

## Slide 5: Framework Landscape (2025)

| Framework | Vibe | Best For | Complexity |
|-----------|------|----------|------------|
| **LangGraph** | Graph-based workflows | Complex multi-step agents | Medium-High |
| **OpenAI Agents SDK** | Official, simple | OpenAI-native apps | Low |
| **Anthropic SDK** | Clean tool use | Claude-native apps | Low |
| **CrewAI** | Role-based teams | Multi-agent collaboration | Medium |
| **AutoGen** | Research-grade | Academic / experimental | High |
| **Pydantic AI** | Type-safe, clean | Production Python apps | Medium |
| **Smolagents** | Lightweight (HuggingFace) | Quick prototypes | Low |

> **SPEAKER NOTES:**
> "Here's the honest take. LangChain got a bad rap for being over-engineered — but LangGraph (its successor for agents) is actually good. OpenAI's Agents SDK is new and simple. CrewAI is fun for multi-agent stuff. Pydantic AI is great if you love type safety. My advice: pick based on your use case, not hype. For learning, we'll use LangGraph because it teaches you the most about how agents work. For production, you might want something simpler."

---

## Slide 6: OpenAI Agents SDK — Simplest Agent

```python
from openai import agents

agent = agents.Agent(
    name="Research Assistant",
    instructions="You help users research topics thoroughly.",
    model="gpt-4o",
    tools=[agents.WebSearchTool()]
)

result = agents.run(agent, "What are the latest AI regulations in the EU?")
print(result.final_output)
```

> **SPEAKER NOTES:**
> "OpenAI released their Agents SDK in early 2025. It's deliberately simple — define an agent, give it tools, run it. Under the hood it's doing the ReAct loop we just saw. The trade-off: it only works with OpenAI models. If you're all-in on OpenAI, this is the fastest path."

---

## Slide 7: Anthropic — Tool Use Agent Pattern

```python
import anthropic

client = anthropic.Anthropic()
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
]

messages = [{"role": "user", "content": "Weather in Toronto and NYC?"}]

# Same ReAct loop, Anthropic style
while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    
    if response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
    else:
        print(response.content[0].text)
        break
```

> **SPEAKER NOTES:**
> "Anthropic doesn't have a dedicated agent SDK — they use their Messages API with tool use. Same ReAct loop, slightly different message format. Notice: Anthropic uses 'tool_use' as a stop reason, OpenAI uses tool_calls on the message. Different packaging, same pattern."

---

## Slide 8: LangGraph — Why It Exists

**LangChain** = chains of LLM calls (linear)
**LangGraph** = graph of nodes with conditional edges (flexible)

```
          ┌─────────┐
          │  START   │
          └────┬─────┘
               │
          ┌────▼─────┐
     ┌────│  Agent   │────┐
     │    └──────────┘    │
     │ (tool call)   (done)│
     │                     │
┌────▼─────┐         ┌────▼─────┐
│  Tools   │         │   END    │
└────┬─────┘         └──────────┘
     │
     └──── (back to Agent)
```

> **SPEAKER NOTES:**
> "LangGraph models your agent as a state machine — a graph. Each node does something (call the LLM, run a tool, check a condition). Edges connect them. This matters when your agent gets complex: maybe it needs human approval before certain actions, or it branches based on the type of request. A while loop can't handle that cleanly. A graph can."

---

## Slide 9: LangGraph — Building an Agent

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

# 1. Define tools
@tool
def search_web(query: str) -> str:
    """Search the web."""
    # real implementation here
    return f"Results for: {query}"

# 2. Define the agent node
llm = ChatOpenAI(model="gpt-4o").bind_tools([search_web])

def agent(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# 3. Define routing logic
def should_continue(state: State):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END

# 4. Build the graph
graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode([search_web]))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, ["tools", END])
graph.add_edge("tools", "agent")

app = graph.compile()

# 5. Run it
result = app.invoke({"messages": [HumanMessage("Research AI regulations")]})
```

> **SPEAKER NOTES:**
> "Here's LangGraph in action. Five steps: define your state, define tools, create the agent node, set up routing logic, build the graph. The key insight is `should_continue` — it checks if the LLM wants to call a tool. If yes, go to the tools node. If no, we're done. The graph compiles into a runnable app. This is more boilerplate than the raw while loop, but it gives you persistence, streaming, human-in-the-loop, and visualization for free."

---

## Slide 10: Pydantic AI — The Type-Safe Option

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class CityInfo(BaseModel):
    name: str
    population: int
    country: str
    fun_fact: str

agent = Agent(
    "openai:gpt-4o",  # or "anthropic:claude-sonnet-4-20250514"
    result_type=CityInfo,  # Structured output built-in
    system_prompt="You provide city information."
)

result = agent.run_sync("Tell me about Tokyo")
print(result.data)  # CityInfo(name='Tokyo', population=13960000, ...)
```

> **SPEAKER NOTES:**
> "Pydantic AI is from the creators of Pydantic — the validation library that powers FastAPI. Their angle: agents should return typed, validated data, not just strings. Notice it's model-agnostic — you can swap 'openai:gpt-4o' for 'anthropic:claude-sonnet' with one line change. If you're building production APIs that need reliable structured output from agents, this is worth a look."

---

## Slide 11: CrewAI — Multi-Agent Teams

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Senior Researcher",
    goal="Find comprehensive info on the topic",
    backstory="You're a thorough researcher who checks multiple sources.",
    llm="gpt-4o"
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear, engaging content",
    backstory="You turn complex research into readable articles.",
    llm="gpt-4o"
)

research_task = Task(
    description="Research the current state of AI agents in 2025",
    agent=researcher,
    expected_output="Detailed research notes"
)

writing_task = Task(
    description="Write a blog post based on the research",
    agent=writer,
    expected_output="A 500-word blog post",
    context=[research_task]  # Gets output from research
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task])
result = crew.kickoff()
```

> **SPEAKER NOTES:**
> "CrewAI takes a different approach — multiple agents with roles that collaborate. The researcher finds information, passes it to the writer. It's intuitive if you think about it like delegating to a team. The catch: you're paying for multiple LLM calls, and sometimes a single well-prompted agent does the same job. Use multi-agent when tasks genuinely require different expertise or perspectives."

---

## Slide 12: Multi-Agent Orchestration Patterns

**Pattern 1: Sequential Pipeline**
```
Agent A → Agent B → Agent C → Result
```
(Like CrewAI example above)

**Pattern 2: Supervisor/Worker**
```
         Supervisor
        /    |     \
    Worker Worker Worker
```
(Supervisor delegates, workers report back)

**Pattern 3: Debate/Consensus**
```
    Agent A ←→ Agent B
         ↓
      Judge Agent
```
(Agents argue, judge picks best answer)

**Pattern 4: Swarm**
```
Agent ←→ Agent ←→ Agent
  ↕         ↕        ↕
Agent ←→ Agent ←→ Agent
```
(OpenAI Swarm pattern — agents hand off to each other)

> **SPEAKER NOTES:**
> "There are four main patterns for multi-agent systems. Sequential is simplest — one agent's output feeds the next. Supervisor/worker is like a manager delegating subtasks. Debate has agents argue and a judge picks the best answer — great for quality. Swarm is OpenAI's pattern where agents can hand off conversations to other agents. Most production systems use sequential or supervisor. Debate and swarm are powerful but expensive."

---

## Slide 13: Agent Memory

| Type | What It Is | How to Implement |
|------|-----------|-----------------|
| **Conversation** | Chat history in the current session | Messages array (you already do this) |
| **Short-term** | Working memory for current task | State/scratchpad in your graph |
| **Long-term** | Persists across sessions | Database, vector store, or file |
| **Episodic** | Past experiences/interactions | Summarized and stored in vector DB |

```python
# LangGraph with persistence (long-term memory)
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()  # or SqliteSaver, PostgresSaver
app = graph.compile(checkpointer=checkpointer)

# Each thread_id gets its own persistent conversation
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"messages": [HumanMessage("Hi!")]}, config)

# Later — same thread picks up where it left off
result = app.invoke({"messages": [HumanMessage("What did I say?")]}, config)
```

> **SPEAKER NOTES:**
> "Memory is what separates a toy agent from a useful one. Conversation memory is just the messages array — you've been doing this since session 1. Short-term memory is the agent's scratchpad during a task. Long-term memory persists across sessions — this is where databases come in. LangGraph has built-in checkpointing that handles this. For production, you'd use PostgresSaver. The episodic memory pattern — where agents remember past interactions — is the frontier right now. OpenAI and Anthropic both added memory features to their consumer products, but for APIs you build it yourself."

---

## Slide 14: Human-in-the-Loop

```python
from langgraph.graph import StateGraph, START, END

def approval_node(state):
    """Pause here and wait for human approval."""
    return state  # LangGraph handles the interrupt

graph = StateGraph(State)
graph.add_node("plan", plan_node)
graph.add_node("approve", approval_node)
graph.add_node("execute", execute_node)

graph.add_edge(START, "plan")
graph.add_edge("plan", "approve")

# Add interrupt BEFORE the execute node
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["execute"]  # ← Pauses here
)

# Run until interrupt
result = app.invoke({"messages": [HumanMessage("Delete old files")]}, config)
# Agent plans, then STOPS before executing

# Human reviews, then resumes
app.invoke(None, config)  # ← Continues from where it paused
```

> **SPEAKER NOTES:**
> "This is critical for production agents. You do NOT want an agent deleting files, sending emails, or making API calls without human approval — at least not yet. LangGraph's interrupt_before lets you pause the graph at any node. The agent plans its action, you review it, then you resume. This is the responsible way to deploy agents. Start with human-in-the-loop for everything, then gradually remove guardrails as you build trust."

---

## Slide 15: When to Use What

| Situation | Recommendation |
|-----------|---------------|
| Simple tool-calling bot | Raw SDK (OpenAI/Anthropic) |
| Complex multi-step workflow | LangGraph |
| Need structured output | Pydantic AI |
| Multi-agent collaboration | CrewAI or LangGraph |
| Quick prototype | OpenAI Agents SDK or Smolagents |
| Production with type safety | Pydantic AI |
| Research / experimental | AutoGen |

**The honest truth:** Most apps don't need agents. A well-crafted prompt with tool use covers 80% of cases.

> **SPEAKER NOTES:**
> "Here's the cut-the-crap moment. Most of you don't need an agent framework. If your app calls an LLM, maybe uses a tool, and returns a result — just use the raw SDK. Frameworks add complexity. Use them when you genuinely need multi-step reasoning, branching logic, persistence, or human-in-the-loop. And if someone tells you to 'just add AI agents' to your product — push back and ask what problem you're actually solving."

---

## Slide 16: Hands-On — Build a Multi-Step Research Agent

We're building a research agent that:
1. Takes a topic
2. Searches for information (simulated)
3. Analyzes what it found
4. Decides if it needs more info
5. Writes a summary

**Open:** `session-5/code/langgraph_agent.py`

> **SPEAKER NOTES:**
> "Time to build. We're creating a research agent with LangGraph. It'll search for info, analyze it, decide if it needs more, and write a final report. This exercises the full ReAct pattern with real decision-making. Open the code file and let's walk through it together."

---

## Slide 17: Recap

✅ Agents = LLMs in a loop with tools and decisions
✅ ReAct = Reason → Act → Observe → Repeat
✅ Frameworks: LangGraph for complex, raw SDK for simple
✅ Memory: conversation → short-term → long-term
✅ Human-in-the-loop: always start with this in production
✅ Don't over-engineer: most apps don't need agents

**Next session:** RAG & Data — teaching your AI about YOUR data

> **SPEAKER NOTES:**
> "Key takeaways: an agent is a loop, not magic. Pick the simplest tool that solves your problem. Always include human oversight in production. Next time, we tackle RAG — how to give your AI access to your own documents and data. That's where things get really practical for enterprise use cases."
