"""
Cut the Crap — Session 1: Multi-Provider Chat Script
Supports OpenAI, Anthropic, and Google Gemini.
Switch providers mid-conversation with /openai, /anthropic, /google.
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

def chat_openai(client, messages: list[dict], model: str = "gpt-4o") -> str:
    """Send messages to OpenAI and return the assistant response."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content


def chat_anthropic(client, messages: list[dict], model: str = "claude-sonnet-4-20250514") -> str:
    """Send messages to Anthropic and return the assistant response.
    
    Note: Anthropic requires max_tokens. System messages are passed separately.
    """
    # Separate system messages from the conversation
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    
    kwargs = dict(
        model=model,
        max_tokens=2048,
        messages=conversation,
    )
    if system_parts:
        kwargs["system"] = "\n".join(system_parts)
    
    response = client.messages.create(**kwargs)
    return response.content[0].text


def chat_google(client, messages: list[dict], model: str = "gemini-2.0-flash") -> str:
    """Send messages to Google Gemini and return the response.
    
    Converts OpenAI-style messages to Gemini format.
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

def chat_openai_stream(client, messages: list[dict], model: str = "gpt-4o") -> str:
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


def chat_anthropic_stream(client, messages: list[dict], model: str = "claude-sonnet-4-20250514") -> str:
    """Stream Anthropic response."""
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    
    kwargs = dict(
        model=model,
        max_tokens=2048,
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


# --- Main Loop ---

PROVIDERS = {
    "openai": {"init": get_openai_client, "chat": chat_openai_stream},
    "anthropic": {"init": get_anthropic_client, "chat": chat_anthropic_stream},
    "google": {"init": get_google_client, "chat": chat_google},  # no stream for simplicity
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
