"""
Cut the Crap — Session 3: Tool-Calling Assistant
A complete assistant with multiple tools, automatic tool loop, and multi-turn.
Uses OpenAI by default. Switch to Anthropic or Google with --provider flag.

Updated: February 2026
Models: GPT-4.1, Claude Sonnet 4.6, Gemini 2.5 Flash

Requirements:
    pip install openai anthropic google-genai
    export OPENAI_API_KEY=your-key
    export ANTHROPIC_API_KEY=your-key
    export GOOGLE_API_KEY=your-key
"""

import json
import math
import sys
from typing import Any

# ============================================================
# Tool Implementations — These are YOUR functions
# ============================================================

def get_weather(city: str, unit: str = "celsius") -> dict:
    """Simulate getting weather data. In production, call a real API."""
    import random
    random.seed(hash(city))
    temp = random.randint(-10, 35) if unit == "celsius" else random.randint(14, 95)
    conditions = ["sunny", "cloudy", "rainy", "snowy", "partly cloudy"]
    return {
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": random.choice(conditions),
        "humidity": random.randint(20, 90),
    }

def search_web(query: str, num_results: int = 3) -> dict:
    """Simulate web search. In production, use a real search API."""
    return {
        "query": query,
        "results": [
            {"title": f"Result {i+1} for '{query}'",
             "url": f"https://example.com/{query.replace(' ', '-')}/{i+1}",
             "snippet": f"This is a relevant result about {query}..."}
            for i in range(num_results)
        ]
    }

def calculate(expression: str) -> dict:
    """Safely evaluate a math expression."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": "Invalid characters in expression. Only numbers and +-*/.(). allowed."}
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


# Registry of available tools
TOOL_REGISTRY: dict[str, callable] = {
    "get_weather": get_weather,
    "search_web": search_web,
    "calculate": calculate,
}

# ============================================================
# Tool Definitions (JSON Schema format)
# ============================================================

TOOL_DEFINITIONS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city. Returns temperature, condition, humidity.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Toronto'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                         "description": "Temperature unit (default: celsius)"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for information. Returns titles, URLs, and snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results (default: 3)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Supports +, -, *, /, parentheses.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string",
                               "description": "Math expression, e.g. '(2 + 3) * 4.5'"},
            },
            "required": ["expression"],
        },
    },
]


def execute_tool(name: str, arguments: dict) -> Any:
    """Look up and execute a tool by name."""
    fn = TOOL_REGISTRY.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**arguments)
    except Exception as e:
        return {"error": f"Tool execution failed: {e}"}


# ============================================================
# OpenAI Tool Loop
# ============================================================

def run_openai(messages: list[dict]):
    """OpenAI tool loop using GPT-4.1.
    
    Pattern: pass tools array, check for tool_calls in response,
    execute tools, append tool results, loop until no more tool_calls.
    """
    from openai import OpenAI
    client = OpenAI()

    tools = [{"type": "function", "function": {
        "name": t["name"],
        "description": t["description"],
        "parameters": t["parameters"],
    }} for t in TOOL_DEFINITIONS]

    while True:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message

        # If no tool calls, we're done
        if not msg.tool_calls:
            return msg.content

        # Process each tool call
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            print(f"  🔧 Calling {call.function.name}({args})")
            result = execute_tool(call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })


# ============================================================
# Anthropic Tool Loop
# ============================================================

def run_anthropic(messages: list[dict]):
    """Anthropic tool loop using Claude Sonnet 4.6.
    
    Pattern: pass tools array, check stop_reason == "tool_use",
    extract tool_use blocks, execute, return tool_result blocks.
    """
    import anthropic
    client = anthropic.Anthropic()

    tools = [{"name": t["name"], "description": t["description"],
              "input_schema": t["parameters"]} for t in TOOL_DEFINITIONS]

    # Extract system message
    system = None
    conversation = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            conversation.append(m)

    while True:
        kwargs = dict(model="claude-sonnet-4-6-20250217", max_tokens=8192,
                      tools=tools, messages=conversation)
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)

        if response.stop_reason != "tool_use":
            # Extract text content
            return next((b.text for b in response.content if b.type == "text"), "")

        # Process tool calls
        conversation.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  🔧 Calling {block.name}({block.input})")
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
        conversation.append({"role": "user", "content": tool_results})


# ============================================================
# Google Tool Loop
# ============================================================

def run_google(messages: list[dict]):
    """Google tool loop using Gemini 2.5 Flash.
    
    Pattern: define FunctionDeclarations in a Tool, check for
    function_call in response parts, return FunctionResponse.
    """
    from google import genai
    from google.genai import types

    client = genai.Client()

    tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=t["name"], description=t["description"],
            parameters=types.Schema(
                type="OBJECT",
                properties={k: types.Schema(type=v.get("type", "STRING").upper(),
                                            description=v.get("description", ""),
                                            enum=v.get("enum"))
                            for k, v in t["parameters"]["properties"].items()},
                required=t["parameters"].get("required", []),
            )
        ) for t in TOOL_DEFINITIONS
    ])

    # Build contents
    system_instruction = None
    contents = []
    for m in messages:
        if m["role"] == "system":
            system_instruction = m["content"]
        else:
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

    config = types.GenerateContentConfig(tools=[tool])
    if system_instruction:
        config.system_instruction = system_instruction

    while True:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=contents, config=config)

        part = response.candidates[0].content.parts[0]
        if not part.function_call:
            return response.text

        fc = part.function_call
        args = dict(fc.args)
        print(f"  🔧 Calling {fc.name}({args})")
        result = execute_tool(fc.name, args)

        contents.append(response.candidates[0].content)
        contents.append(types.Content(parts=[
            types.Part(function_response=types.FunctionResponse(
                name=fc.name, response=result))
        ]))


# ============================================================
# Main Chat Loop
# ============================================================

RUNNERS = {
    "openai": run_openai,
    "anthropic": run_anthropic,
    "google": run_google,
}

def main():
    provider = "openai"
    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        if idx + 1 < len(sys.argv):
            provider = sys.argv[idx + 1]

    if provider not in RUNNERS:
        print(f"Unknown provider: {provider}. Use: openai, anthropic, google")
        sys.exit(1)

    run_fn = RUNNERS[provider]

    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful assistant with access to tools. "
            "Use tools when needed to answer questions accurately. "
            "Always prefer using the calculate tool for math instead of doing it yourself. "
            "Be concise and direct."
        ),
    }
    history = [system_prompt]

    print("=" * 60)
    print(f"  Tool-Calling Assistant ({provider})")
    print(f"  Tools: get_weather, search_web, calculate")
    print(f"  Type /quit to exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            print("Bye!")
            break

        history.append({"role": "user", "content": user_input})

        try:
            response_text = run_fn(history)
            print(f"\nAssistant: {response_text}")
            history.append({"role": "assistant", "content": response_text})
        except Exception as e:
            print(f"\n❌ Error: {e}")
            history.pop()  # remove failed user message


if __name__ == "__main__":
    main()
