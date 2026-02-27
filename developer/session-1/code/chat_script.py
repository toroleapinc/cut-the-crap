"""
Cut the Crap — Session 1: Multi-Provider Chat Script
Supports OpenAI, Anthropic, and Google Gemini.
Switch providers mid-conversation with /openai, /anthropic, /google.

Updated: February 2026
Models: GPT-4.1, Claude Sonnet 4.6, Gemini 2.5 Flash

Requirements:
    pip install openai anthropic google-genai
    export OPENAI_API_KEY=your-key
    export ANTHROPIC_API_KEY=your-key
    export GOOGLE_API_KEY=your-key
"""

import os
import sys

# --- Provider Clients ---

def get_openai_client():
    from openai import OpenAI
    return OpenAI()

def get_anthropic_client():
    import anthropic
    return anthropic.Anthropic()

def get_google_client():
    from google import genai
    return genai.Client()


# --- Chat Functions ---

def chat_openai(client, messages: list[dict], model: str = "gpt-4.1") -> str:
    """Send messages to OpenAI and return the assistant response.
    
    Models available (Feb 2026):
      - gpt-5       : Most capable reasoning model
      - gpt-5-mini  : Fast reasoning model
      - gpt-4.1     : Best non-reasoning model (default)
      - gpt-4.1-mini: Smaller, faster version
      - gpt-4.1-nano: Fastest, cheapest
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content


def chat_anthropic(client, messages: list[dict], model: str = "claude-sonnet-4-6-20250217") -> str:
    """Send messages to Anthropic and return the assistant response.
    
    Note: Anthropic requires max_tokens. System messages are passed separately.
    
    Models available (Feb 2026):
      - claude-opus-4-6-20250217   : Most capable
      - claude-sonnet-4-6-20250217 : Best balance (default)
      - claude-haiku-3-5-20241022  : Fastest, cheapest
    """
    # Separate system messages from the conversation
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    
    kwargs = dict(
        model=model,
        max_tokens=8192,
        messages=conversation,
    )
    if system_parts:
        kwargs["system"] = "\n".join(system_parts)
    
    response = client.messages.create(**kwargs)
    return response.content[0].text


def chat_google(client, messages: list[dict], model: str = "gemini-2.5-flash") -> str:
    """Send messages to Google Gemini and return the response.
    
    Converts OpenAI-style messages to Gemini format.
    
    Models available (Feb 2026):
      - gemini-2.5-pro  : Most capable
      - gemini-2.5-flash: Fast and efficient (default)
    """
    # Extract system instruction
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    
    # Convert to Gemini content format
    contents = []
    for msg in conversation:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    
    config = {}
    if system_parts:
        config["system_instruction"] = "\n".join(system_parts)
    
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config if config else None,
    )
    return response.text


# --- Streaming Variants ---

def chat_openai_stream(client, messages: list[dict], model: str = "gpt-4.1") -> str:
    """Stream OpenAI response, printing tokens as they arrive."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        stream=True,
    )
    full_response = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_response.append(delta)
    print()  # newline after stream
    return "".join(full_response)


def chat_anthropic_stream(client, messages: list[dict], model: str = "claude-sonnet-4-6-20250217") -> str:
    """Stream Anthropic response."""
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    
    kwargs = dict(
        model=model,
        max_tokens=8192,
        messages=conversation,
    )
    if system_parts:
        kwargs["system"] = "\n".join(system_parts)
    
    full_response = []
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response.append(text)
    print()
    return "".join(full_response)


def chat_google_stream(client, messages: list[dict], model: str = "gemini-2.5-flash") -> str:
    """Stream Google Gemini response."""
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    
    contents = []
    for msg in conversation:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    
    config = {}
    if system_parts:
        config["system_instruction"] = "\n".join(system_parts)
    
    full_response = []
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config if config else None,
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)
            full_response.append(chunk.text)
    print()
    return "".join(full_response)


# --- Main Loop ---

PROVIDERS = {
    "openai": {"init": get_openai_client, "chat": chat_openai_stream},
    "anthropic": {"init": get_anthropic_client, "chat": chat_anthropic_stream},
    "google": {"init": get_google_client, "chat": chat_google_stream},
}

def main():
    provider_name = "openai"
    clients = {}
    
    # System prompt — same across all providers
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant. Be concise and direct."
    }
    
    # Conversation history (OpenAI message format as lingua franca)
    history: list[dict] = [system_message]
    
    print("=" * 60)
    print("  Cut the Crap — Multi-Provider Chat")
    print("  Commands: /openai /anthropic /google /clear /quit")
    print(f"  Current provider: {provider_name}")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/quit":
                print("Bye!")
                break
            elif cmd == "/clear":
                history = [system_message]
                print("🗑️  History cleared.")
                continue
            elif cmd.lstrip("/") in PROVIDERS:
                provider_name = cmd.lstrip("/")
                print(f"🔄 Switched to {provider_name}")
                continue
            else:
                print(f"Unknown command: {cmd}")
                continue
        
        # Add user message to history
        history.append({"role": "user", "content": user_input})
        
        # Initialize client lazily
        if provider_name not in clients:
            try:
                clients[provider_name] = PROVIDERS[provider_name]["init"]()
            except Exception as e:
                print(f"❌ Failed to init {provider_name}: {e}")
                history.pop()  # remove the user message
                continue
        
        # Call the provider
        client = clients[provider_name]
        chat_fn = PROVIDERS[provider_name]["chat"]
        
        print(f"\n[{provider_name}]: ", end="", flush=True)
        try:
            response_text = chat_fn(client, history)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            history.pop()
            continue
        
        # Add assistant response to history
        history.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
