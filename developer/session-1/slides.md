# Session 1: The AI Landscape & APIs
## Cut the Crap — AI Engineer Edition

---

## Slide 1: Welcome & What This Course Is

**SHOW:**
```
Cut the Crap — AI Engineer Edition
Session 1 of 8: The AI Landscape & APIs

You use ChatGPT. Cool.
Now let's learn to BUILD with AI.
```

**SAY:**
> Welcome to Cut the Crap. This course is for developers who use ChatGPT daily but haven't built with AI APIs yet. Over 8 sessions, you'll go from "I paste stuff into ChatGPT" to "I build AI-powered applications." No fluff, no hype — just practical engineering. By the end of today, you'll have made your first API calls to multiple AI providers.

---

## Slide 2: The AI Provider Landscape (Topic 1)

**SHOW:**

| Provider | Key Models | Strengths | Pricing Tier |
|----------|-----------|-----------|-------------|
| **OpenAI** | GPT-5.2 (Dec 2025), 400K ctx, o3, o4-mini | Ecosystem, MCP support, 187 tok/s | $20/$60 per 1M |
| **Anthropic** | Claude Opus 4.6 (Feb 2026), Sonnet 4.6, Haiku 4.5 | SWE-bench 80.9%, Claude Code, safety | $5/$25 per 1M |
| **Google** | Gemini 3 Pro (Nov 2025), 1M ctx | Native multimodal (text+image+audio+video), agentic | $-$$ |
| **Meta** | Llama 4 Scout/Maverick (open source) | 17B active MoE, self-hostable, commercial OK | Free (compute) |
| **Mistral** | Mistral Large, Codestral | EU-based, efficient, open-weight options | $-$$ |
| **DeepSeek** | DeepSeek V4 ($0.14/1M tokens) | 94% cheaper than GPT, strong reasoning | Budget king |

**SAY:**
> Here's the landscape as of early 2026. Six major players. OpenAI has market share and the biggest ecosystem. Anthropic — that's Claude — excels at safety, long context, and coding tasks. Google's Gemini has insane context windows, over a million tokens. Meta's Llama is the open-source king — you can run it yourself. Mistral is the EU player, efficient models. DeepSeek out of China shocked everyone with strong models at rock-bottom prices.
>
> Key takeaway: there is NO single best provider. It depends on your use case, budget, and constraints. This course covers all of them fairly.

---

## Slide 3: Models Deep Dive — Types (Topic 2)

**SHOW:**
```
Model Types:

1. CHAT MODELS (most common)
   - GPT-5.2, Claude Sonnet 4.6, Gemini 3 Pro
   - Input: messages → Output: text
   - What you use 95% of the time

2. REASONING MODELS
   - o3, o4-mini (OpenAI), Claude Opus 4.6 (extended thinking)
   - "Think before answering" — chain-of-thought built in
   - Slower, more expensive, better at hard problems

3. EMBEDDING MODELS
   - text-embedding-3-small (OpenAI), Gemini embedding
   - Input: text → Output: vector of numbers
   - Used for search, RAG (Session 6)

4. IMAGE/AUDIO MODELS
   - DALL-E 3, Imagen 3, Whisper, TTS
   - Specialized — we cover these in Session 2
```

**SAY:**
> Models aren't all the same. Chat models are your bread and butter — you send messages, you get text back. Reasoning models are newer — they actually "think" before responding. They're slower and cost more, but they crush math, logic, and complex coding problems. Embedding models turn text into numbers — vectors — which is how you do semantic search. We'll use those in the RAG session. And then there are specialized models for images and audio.

---

## Slide 4: Key Parameters

**SHOW:**
```python
response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[...],
    temperature=0.7,      # 0 = deterministic, 2 = creative chaos
    max_tokens=1024,       # output length cap
    top_p=0.9,             # nucleus sampling (usually leave at 1)
    frequency_penalty=0,   # reduce repetition
    presence_penalty=0,    # encourage new topics
    stop=["\n\n"],         # stop sequences
)
```

```
Temperature Guide:
  0.0  → Code generation, data extraction, deterministic tasks
  0.3  → Customer support, summarization
  0.7  → General chat, creative assistance (DEFAULT)
  1.0+ → Brainstorming, creative writing
```

**SAY:**
> Temperature is the most important parameter. Zero means the model gives you the same answer every time — use that for code and data extraction. 0.7 is the default sweet spot. Above 1.0 gets wild. Max tokens caps the output length — not the input. Top_p is another way to control randomness — generally leave it alone if you're setting temperature. The stop parameter lets you tell the model to halt when it hits a specific string.

---

## Slide 5: Open Source Models (Topic 3)

**SHOW:**
```
Open Source / Open Weight Models:

┌─────────────────────────────────────────────────┐
│  Llama 4 (Meta)                                  │
│  - Scout (17B active, 109B total, 16 experts)    │
│  - Maverick (17B active, 400B total, 128 experts)│
│  - Open source, commercial use OK                │
├─────────────────────────────────────────────────┤
│  DeepSeek V4 / R1                                │
│  - V3: 671B MoE, strong general model            │
│  - R1: reasoning model, open weights             │
│  - MIT license                                   │
├─────────────────────────────────────────────────┤
│  Mistral / Codestral                             │
│  - Small (24B), Large (123B)                     │
│  - Codestral for code tasks                      │
│  - Apache 2.0 for small models                   │
├─────────────────────────────────────────────────┤
│  Qwen 2.5 (Alibaba)                             │
│  - 0.5B to 72B sizes                             │
│  - Strong multilingual                           │
│  - Apache 2.0                                    │
└─────────────────────────────────────────────────┘
```

**SAY:**
> Open source is a huge deal. Meta's Llama 4 uses mixture-of-experts — only 17 billion parameters are active at once, but the full model is much bigger. DeepSeek R1 is open weights and gives you reasoning capabilities for free. Mistral and Qwen round out the options. Why does this matter? You can run these yourself, fine-tune them, or use them where you can't send data to external APIs. We'll cover self-hosting next.

---

## Slide 6: Self-Hosting — Ollama & vLLM (Topic 4)

**SHOW:**
```bash
# Ollama — dead simple local inference
$ curl -fsSL https://ollama.ai/install.sh | sh
$ ollama pull llama4
$ ollama run llama4
>>> Hello! How can I help?

# Ollama exposes an OpenAI-compatible API!
$ curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama4", "messages": [{"role": "user", "content": "Hello"}]}'

# vLLM — production-grade serving
$ pip install vllm
$ vllm serve meta-llama/Llama-4-Scout \
    --tensor-parallel-size 4   # multi-GPU
```

```
When to self-host:
✅ Data can't leave your network
✅ You need to fine-tune
✅ High-volume, cost-sensitive
✅ Low-latency edge deployment

When NOT to self-host:
❌ You need frontier intelligence (GPT-5.2, Claude Opus 4.6 level)
❌ Small team, no GPU budget
❌ Rapid prototyping
```

**SAY:**
> Ollama is the Docker of local AI. One command, you're running Llama on your laptop. The killer feature: it exposes an OpenAI-compatible API, so your code works with zero changes. vLLM is for production — it handles batching, multi-GPU, and high throughput. Self-host when you have data constraints or high volume. Don't self-host if you need the absolute best intelligence — frontier models from OpenAI and Anthropic are still ahead of what you can run locally.

---

## Slide 7: API Key Setup — Live First Call (Topic 5)

**SHOW:**
```bash
# Get your API keys:
# OpenAI:    https://platform.openai.com/api-keys
# Anthropic: https://console.anthropic.com/settings/keys
# Google:    https://aistudio.google.com/apikey

# Set as environment variables (never hardcode!)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AI..."

# Install SDKs
pip install openai anthropic google-genai
```

```python
# === OPENAI — First Call ===
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from env

response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)
print(response.choices[0].message.content)
```

```python
# === ANTHROPIC — First Call ===
import anthropic
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-sonnet-4-6-20250217",
    max_tokens=100,
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)
print(response.content[0].text)
```

```python
# === GOOGLE — First Call ===
from google import genai
client = genai.Client()  # reads GOOGLE_API_KEY from env

response = client.models.generate_content(
    model="gemini-3-pro",
    contents="Say hello in one sentence."
)
print(response.text)
```

**SAY:**
> Let's get our hands dirty. Everyone go to these three URLs and grab API keys. OpenAI gives you $5 free credit. Anthropic gives you $5. Google's Gemini API has a generous free tier. Set them as environment variables — NEVER put keys in your code.
>
> Notice the APIs are slightly different. OpenAI uses `chat.completions.create`. Anthropic uses `messages.create` and requires `max_tokens`. Google's is `generate_content`. Different shapes, same idea: send text, get text back. The SDKs all auto-read their key from the environment.

---

## Slide 8: OAuth vs API Key (Topic 6)

**SHOW:**
```
API Key:
  ✓ Simple — one string, paste it, go
  ✓ Good for: server-to-server, personal projects, backends
  ✗ Can't scope to users, can't rotate easily in production
  ✗ If leaked, full access until revoked

OAuth 2.0:
  ✓ Per-user auth, scoped permissions, token refresh
  ✓ Good for: multi-user apps, marketplace integrations
  ✗ Complex setup (redirect URIs, token management)
  ✗ Not all AI providers support it yet

In practice:
  → Prototyping & learning: API keys
  → Production multi-user app: OAuth (where available)
  → Most AI APIs today: API keys with per-key rate limits
```

**SAY:**
> Quick note on auth. API keys are what you'll use 90% of the time during this course and for most backend integrations. OAuth matters when you're building something where end users authenticate — like a ChatGPT plugin or a marketplace skill. OpenAI's GPT Store uses OAuth for some integrations. For now, API keys are fine. Just keep them out of git — use `.env` files or secrets managers.

---

## Slide 9: API Comparison — Side by Side

**SHOW:**

| Feature | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| Endpoint | `chat.completions.create` | `messages.create` | `generate_content` |
| Auth | `OPENAI_API_KEY` | `ANTHROPIC_API_KEY` | `GOOGLE_API_KEY` |
| System prompt | `{"role": "system", ...}` | `system=` parameter | `system_instruction=` |
| Max tokens | Optional (defaults vary) | **Required** | Optional |
| Streaming | `stream=True` | `stream=True` via `.stream()` | `stream=True` via `.stream()` |
| Response | `choices[0].message.content` | `content[0].text` | `.text` |
| Pricing unit | per 1K tokens | per 1K tokens | per 1K tokens (free tier!) |

**SAY:**
> Here's your cheat sheet. Three providers, three slightly different APIs. The biggest gotcha: Anthropic requires max_tokens — it won't default for you. System prompts go in different places — OpenAI puts it in the messages array, Anthropic and Google have separate parameters. Bookmark this slide — you'll reference it constantly.

---

## Slide 10: Hands-On — Build a Chat Script (Topic 7)

**SHOW:**
```
📝 Exercise: Build a multi-provider chat script

Requirements:
1. Accept user input in a loop
2. Support OpenAI, Anthropic, and Google
3. Maintain conversation history
4. Switch providers with a command (/openai, /anthropic, /google)
5. Type /quit to exit

Time: 20 minutes
Starter code: session-1/code/chat_script.py
```

**SAY:**
> Time to build. Open `chat_script.py` — it's a complete, working multi-provider chat script. Walk through it, run it, modify it. The key thing to understand is conversation history — every API call sends the FULL conversation so far. The model has no memory between calls. You manage the context. This is the most fundamental thing to understand about AI APIs.

---

## Slide 11: How Conversation History Works

**SHOW:**
```python
# Call 1:
messages = [
    {"role": "user", "content": "My name is Alice"}
]
# Model responds: "Hi Alice!"

# Call 2 — you send EVERYTHING again:
messages = [
    {"role": "user", "content": "My name is Alice"},
    {"role": "assistant", "content": "Hi Alice!"},
    {"role": "user", "content": "What's my name?"}
]
# Model responds: "Your name is Alice!"

# WITHOUT history:
messages = [
    {"role": "user", "content": "What's my name?"}
]
# Model responds: "I don't know your name."
```

```
Key insight: The API is STATELESS.
ChatGPT maintains history for you.
The API does NOT. You send it every time.
Each call costs tokens for the ENTIRE conversation.
```

**SAY:**
> This trips up everyone. When you use ChatGPT, it remembers your conversation. The API doesn't. Every single API call, you send the full conversation history. The model reads it all, generates a response, and forgets everything. Next call, you send it all again plus the new message. This means conversations get more expensive over time as the context grows. This is why context window size matters — it's the max conversation length the model can handle.

---

## Slide 12: Streaming

**SHOW:**
```python
# OpenAI streaming
stream = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    stream=True
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)

# Anthropic streaming
with anthropic_client.messages.stream(
    model="claude-sonnet-4-6-20250217",
    max_tokens=200,
    messages=[{"role": "user", "content": "Tell me a joke"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

```
Why stream?
- Users see output immediately (perceived speed)
- Time to first token: ~200ms vs waiting 2-5s for full response
- Essential for any user-facing application
```

**SAY:**
> Streaming gives you the ChatGPT-like typing effect. Without streaming, you wait for the entire response. With streaming, tokens arrive as they're generated. The APIs differ slightly — OpenAI uses `stream=True` and iterates chunks, Anthropic has a context manager with `text_stream`. Always stream in user-facing apps. The time-to-first-token is what users perceive as "fast."

---

## Slide 13: Error Handling & Rate Limits

**SHOW:**
```python
from openai import OpenAI, RateLimitError, APIError
import time

client = OpenAI()

def call_with_retry(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-5.2",
                messages=messages
            )
        except RateLimitError:
            wait = 2 ** attempt  # exponential backoff
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
        except APIError as e:
            print(f"API error: {e}")
            raise
    raise Exception("Max retries exceeded")
```

```
Common HTTP errors:
  401 — Bad API key
  429 — Rate limited (too many requests)
  500 — Provider is down
  529 — Provider overloaded (Anthropic)
```

**SAY:**
> You will get rate limited. It's not a question of if. Exponential backoff is the standard pattern — wait 1 second, then 2, then 4. All three providers have rate limits based on tokens per minute and requests per minute. Pro tip: the SDKs have built-in retry logic you can configure, but understanding the pattern matters.

---

## Slide 14: Session 1 Recap

**SHOW:**
```
✅ 6 major AI providers — no single winner
✅ Model types: chat, reasoning, embedding, specialized
✅ Temperature controls creativity vs determinism
✅ Open source: Llama, DeepSeek, Mistral, Qwen
✅ Self-hosting: Ollama (easy) / vLLM (production)
✅ API calls: OpenAI, Anthropic, Google — same concept, different shapes
✅ Conversation history is YOUR responsibility
✅ Always stream in user-facing apps
✅ API keys in env vars, never in code

Next session: Prompt engineering, structured output, multimodal
```

**SAY:**
> That's Session 1. You now know the landscape, you've made API calls to three providers, and you've built a working chat script. The most important thing to remember: the API is stateless, you manage the history, and there's no single best provider. Next session we'll level up with prompt engineering, structured output (getting JSON back reliably), and multimodal — vision, audio, images. See you then.
