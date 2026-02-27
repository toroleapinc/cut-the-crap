# Session 3: Tool Use & Custom Assistants
## Cut the Crap — AI Engineer Edition

---

## Slide 1: Why Tool Use Matters (Topic 12)

**SHOW:**
```
Without tools:               With tools:
┌─────────────┐              ┌─────────────┐
│   LLM       │              │   LLM       │
│             │              │   ↕ calls    │
│  "I can't   │              │  functions   │
│   access    │              │   ↕          │
│   real data"│              │  Real World  │
└─────────────┘              └─────────────┘

Tools let LLMs:
✅ Get real-time data (weather, stock prices, databases)
✅ Take actions (send emails, create files, update records)
✅ Use external APIs (any API becomes an AI capability)
✅ Do math reliably (calculator tool > LLM arithmetic)
```

**SAY:**
> Until now, the LLM just generates text. It can't check the weather, query your database, or send an email. Tool use changes everything. You define functions the model CAN call, the model decides WHEN to call them, and your code EXECUTES them. The model never runs code itself — it asks you to run it and gives you the arguments. This is the bridge from "chatbot" to "AI that does things."

---

## Slide 2: How Tool Use Works — The Loop

**SHOW:**
```
The Tool Use Loop:

1. You → Model: "What's the weather in Toronto?"
   (with tool definitions attached)

2. Model → You: "I want to call get_weather(city='Toronto')"
   (tool_call response, NOT a text response)

3. You: Execute get_weather("Toronto") → {"temp": -5, "condition": "snow"}

4. You → Model: "Here's the result: {temp: -5, condition: snow}"
   (tool_result message)

5. Model → You: "It's -5°C and snowing in Toronto. Bundle up!"
   (final text response to user)

Key: The MODEL chooses what to call. YOUR CODE executes it.
The model NEVER runs code directly.
```

**SAY:**
> Here's the flow. You send the user's question along with a list of available tools. The model looks at the question, decides it needs the weather tool, and returns a structured tool call — not text. Your code catches that, runs the actual function, and sends the result back. Then the model uses the result to write a human-friendly response. Two round trips. The model is the brain, your code is the hands.

---

## Slide 3: OpenAI Function Calling

**SHOW:**
```python
from openai import OpenAI
import json

client = OpenAI()

# Step 1: Define tools
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                         "description": "Temperature unit"}
            },
            "required": ["city"]
        }
    }
}]

# Step 2: Send message with tools
response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "What's the weather in Toronto?"}],
    tools=tools,
)

# Step 3: Check if model wants to call a tool
msg = response.choices[0].message
if msg.tool_calls:
    call = msg.tool_calls[0]
    args = json.loads(call.function.arguments)
    print(f"Model wants to call: {call.function.name}({args})")
    
    # Step 4: Execute and send result back
    result = get_weather(**args)  # YOUR function
    
    followup = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "user", "content": "What's the weather in Toronto?"},
            msg,  # include the assistant's tool_call message
            {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)},
        ],
        tools=tools,
    )
    print(followup.choices[0].message.content)
```

**SAY:**
> OpenAI's approach. You define tools as JSON schemas. The model returns `tool_calls` instead of content. You parse the arguments, execute your function, and send the result back in a `tool` role message. Note the `tool_call_id` — it links the result to the specific call. The model can call multiple tools in one response.

---

## Slide 4: Anthropic Tool Use

**SHOW:**
```python
import anthropic
import json

client = anthropic.Anthropic()

# Define tools (same concept, different shape)
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["city"]
    }
}]

# Send message
response = client.messages.create(
    model="claude-sonnet-4-6-20250217",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Toronto?"}]
)

# Check for tool use
if response.stop_reason == "tool_use":
    tool_block = next(b for b in response.content if b.type == "tool_use")
    print(f"Calling: {tool_block.name}({tool_block.input})")
    
    result = get_weather(**tool_block.input)
    
    # Send result back
    followup = client.messages.create(
        model="claude-sonnet-4-6-20250217",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "What's the weather in Toronto?"},
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": json.dumps(result),
            }]},
        ],
    )
    # Extract text from response
    text = next(b.text for b in followup.content if b.type == "text")
    print(text)
```

**SAY:**
> Anthropic's tool use is similar but the shapes differ. Tools use `input_schema` instead of `parameters`. The response has content blocks — you look for `tool_use` type blocks. Results go back as `tool_result` blocks in a user message. The `stop_reason` tells you why the model stopped — "tool_use" means it wants to call something, "end_turn" means it's done talking.

---

## Slide 5: Google Tool Use

**SHOW:**
```python
from google import genai
from google.genai import types

client = genai.Client()

# Define tools
weather_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Get current weather for a city",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "city": types.Schema(type="STRING", description="City name"),
                    "unit": types.Schema(
                        type="STRING", enum=["celsius", "fahrenheit"]
                    ),
                },
                required=["city"],
            ),
        )
    ]
)

response = client.models.generate_content(
    model="gemini-3-pro",
    contents="What's the weather in Toronto?",
    config=types.GenerateContentConfig(tools=[weather_tool]),
)

# Check for function call
part = response.candidates[0].content.parts[0]
if part.function_call:
    fc = part.function_call
    print(f"Calling: {fc.name}({dict(fc.args)})")
    
    result = get_weather(**dict(fc.args))
    
    # Send result back
    followup = client.models.generate_content(
        model="gemini-3-pro",
        contents=[
            types.Content(role="user", parts=[types.Part(text="What's the weather in Toronto?")]),
            response.candidates[0].content,  # assistant's function call
            types.Content(parts=[types.Part(function_response=types.FunctionResponse(
                name=fc.name, response=result
            ))]),
        ],
        config=types.GenerateContentConfig(tools=[weather_tool]),
    )
    print(followup.text)
```

**SAY:**
> Google's is more verbose but same pattern. They use their own `types` module for schemas. The response has `function_call` parts. You send back a `FunctionResponse`. Honestly, Google's SDK is the most verbose of the three, but it works fine. In practice, you'd probably wrap all three behind a common interface.

---

## Slide 6: Tool Use Comparison

**SHOW:**

| Feature | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| Tool definition | `tools=[{"type": "function", ...}]` | `tools=[{"name": ..., "input_schema": ...}]` | `types.Tool(function_declarations=...)` |
| Schema location | `function.parameters` | `input_schema` | `FunctionDeclaration.parameters` |
| Detection | `msg.tool_calls` | `stop_reason == "tool_use"` | `part.function_call` |
| Arguments | `json.loads(call.function.arguments)` | `tool_block.input` (already dict) | `dict(fc.args)` |
| Result role | `"tool"` | `"user"` with `tool_result` block | `FunctionResponse` part |
| Parallel calls | ✅ Yes | ✅ Yes | ✅ Yes |
| Force tool | `tool_choice={"type": "function", "function": {"name": "..."}}` | `tool_choice={"type": "tool", "name": "..."}` | `tool_config={"function_calling_config": {"mode": "ANY"}}` |

**SAY:**
> Your cheat sheet. The concepts are identical, the spellings differ. One nice thing about Anthropic: the arguments come as a Python dict already, no JSON parsing. All three support parallel tool calls — the model can call multiple tools at once. And all three let you force a specific tool with `tool_choice`.

---

## Slide 7: Assistants API vs Messages API (Topic 13)

**SHOW:**
```
OpenAI Assistants API:
┌────────────────────────────────────────┐
│  Server-side state management          │
│  - Threads (conversations)             │
│  - Runs (execution)                    │
│  - Built-in: code interpreter,         │
│    file search, function calling       │
│  - Automatic conversation memory       │
│  - File uploads attached to threads    │
│  Pro: Less code for stateful apps      │
│  Con: Vendor lock-in, less control     │
└────────────────────────────────────────┘

Anthropic / Google Messages API:
┌────────────────────────────────────────┐
│  Stateless — you manage everything     │
│  - Send messages, get response         │
│  - You store conversation history      │
│  - You manage files, context, memory   │
│  Pro: Full control, portable           │
│  Con: More code to write               │
└────────────────────────────────────────┘

Recommendation: Start with Messages API (stateless).
Use Assistants when you need server-side features
(code interpreter, persistent file search).
```

**SAY:**
> OpenAI's Assistants API manages state server-side. You create a "thread," add messages, and "run" the assistant. It remembers context, can search files, and execute code — all on OpenAI's servers. Anthropic and Google keep it simple: stateless messages in, messages out. I'd recommend starting stateless. You understand exactly what's happening, it's portable across providers, and you're not locked in. Use Assistants when you specifically need code interpreter or built-in file search.

---

## Slide 8: Custom GPTs vs Skills vs Gems (Topic 14)

**SHOW:**
```
Custom GPTs (OpenAI):
  - Build via ChatGPT UI — no code needed
  - Custom instructions, knowledge files, API actions
  - Publish to GPT Store
  - Great for non-developers, limited for devs

Claude Projects (Anthropic):
  - Project-level system prompts and knowledge
  - Upload files as context
  - No marketplace (yet)

Google Gems (Gemini):
  - Custom personas in Gemini
  - Instructions + context
  - Limited distribution

OpenClaw Skills:
  - Code-based, version controlled
  - Full tool access (MCP, APIs)
  - ClawHub marketplace
  - Most flexible but requires coding
```

**SAY:**
> Each platform has their version of "customizable AI." GPTs are the most mature — there's a whole store. Claude has Projects for team-level customization. Gems is Google's entry. These are all consumer/prosumer tools. For developers, you'll want more control — which is where custom tool-calling assistants (what we built today) and skills platforms like OpenClaw come in. Session 8 covers OpenClaw in depth.

---

## Slide 9: Building a Complete Tool-Calling Assistant (Topic 15)

**SHOW:**
```
📝 Exercise: Build a tool-calling assistant

Your assistant should:
1. Have a personality (system prompt)
2. Support 3+ tools:
   - get_weather(city) → weather data
   - search_web(query) → search results  
   - calculate(expression) → math result
3. Handle the full tool loop automatically
4. Support multi-turn conversation
5. Handle multiple tool calls in one turn

Time: 30 minutes
Starter code: session-3/code/tool_calling.py
```

**SAY:**
> Let's build it. The starter code has a complete tool-calling assistant with three tools, automatic loop handling, and multi-turn support. It handles the full cycle: user asks something, model calls tools, code executes them, results go back, model responds. Walk through it, run it, add your own tools.

---

## Slide 10: Session 3 Recap

**SHOW:**
```
✅ Tool use: model decides WHAT to call, your code EXECUTES
✅ The loop: message → tool_call → execute → tool_result → response
✅ All 3 providers: same concept, different JSON shapes
✅ Parallel tool calls: model can call multiple tools at once
✅ Assistants API: server-side state (OpenAI only)
✅ Messages API: stateless, portable, recommended starting point
✅ Custom GPTs/Projects/Gems: consumer-grade customization
✅ Production: always validate tool arguments, handle errors

Next session: MCP — the universal tool protocol
```

**SAY:**
> Session 3 done. You've built an AI that can actually DO things — check weather, search the web, do math. The tool use pattern is THE fundamental building block of agentic AI. Everything in the next sessions builds on this loop. Next session: MCP — Model Context Protocol — which standardizes how tools connect to AI, so you don't have to write custom integrations for everything.
