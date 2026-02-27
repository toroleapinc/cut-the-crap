# Session 8: Production, Dev Tools & OpenClaw
## Cut the Crap — AI Engineer Edition

---

## Slide 1: What We're Covering Today

- Cost optimization: caching, batching, model routing
- AI gateways: LiteLLM, Portkey, OpenRouter
- Production patterns: retries, streaming, error handling
- Coding tools: Claude Code vs Cursor vs Copilot
- Privacy & enterprise concerns
- OpenClaw deep dive: skills, agents, ClawHub
- **Hands-on:** Build your own OpenClaw skill
- The AI engineer toolkit map

> **SPEAKER NOTES:**
> "Final session. We've built agents, RAG pipelines, added tracing and evals. Now we ship it. Today is about the engineering that makes AI actually work in production — cost control, reliability, tooling, and the platform layer. Plus we'll build an OpenClaw skill and look at the full picture of what an AI engineer needs to know."

---

## Slide 2: Cost Optimization — It Adds Up Fast

**Real costs at scale (GPT-5.2):**
| Usage | Input Tokens | Output Tokens | Monthly Cost |
|-------|-------------|---------------|-------------|
| 1K requests/day | ~5M/day | ~1M/day | ~$1,050 |
| 10K requests/day | ~50M/day | ~10M/day | ~$10,500 |
| 100K requests/day | ~500M/day | ~100M/day | ~$105,000 |

**Three levers:**
1. **Use cheaper models** where possible (GPT-5.2-mini is 30x cheaper)
2. **Cache** repeated/similar requests
3. **Batch** non-urgent requests (50% discount)

> **SPEAKER NOTES:**
> "AI costs can explode. A single GPT-5.2 call costs fractions of a cent, but at 100K requests per day you're looking at $100K/month. The good news: most of that is waste. You probably don't need GPT-5.2 for every request. Caching saves repeat queries. And batching gets you a 50% discount for anything that doesn't need real-time response."

---

## Slide 3: Model Routing — The Biggest Win

```python
def route_request(query: str, complexity: str = "auto") -> str:
    """Route to the right model based on task complexity."""
    
    if complexity == "auto":
        # Use a cheap model to classify complexity
        classification = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": 
                f"Rate this query's complexity: LOW, MEDIUM, or HIGH.\n{query}"}],
            max_tokens=10, temperature=0
        ).choices[0].message.content
        complexity = classification.strip()
    
    model_map = {
        "LOW": "gpt-4o",       # $0.15/$0.60 per 1M tokens
        "MEDIUM": "gpt-5.2",          # $2.50/$10 per 1M tokens  
        "HIGH": "o1",                # For complex reasoning
    }
    
    model = model_map.get(complexity, "gpt-5.2")
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content
```

**Real-world split:** ~70% LOW, ~25% MEDIUM, ~5% HIGH → **60-80% cost savings**

> **SPEAKER NOTES:**
> "Model routing is the single biggest cost optimization. Most requests don't need your most expensive model. 'What's the weather?' doesn't need GPT-5.2. 'Analyze this legal contract for liability issues' does. Use a cheap classifier to route — the cost of classification is negligible compared to the savings. Companies that implement routing typically save 60-80% on their AI costs."

---

## Slide 4: Caching Strategies

```python
# 1. Exact match cache (simplest)
import hashlib, json, redis

r = redis.Redis()

def cached_llm_call(messages: list, model: str = "gpt-5.2") -> str:
    cache_key = hashlib.md5(json.dumps(messages).encode()).hexdigest()
    
    cached = r.get(cache_key)
    if cached:
        return cached.decode()
    
    response = client.chat.completions.create(model=model, messages=messages)
    result = response.choices[0].message.content
    
    r.setex(cache_key, 3600, result)  # Cache for 1 hour
    return result

# 2. Anthropic Prompt Caching (built-in, no extra infra)
response = anthropic_client.messages.create(
    model="claude-sonnet-4-6-20250217",
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": very_long_system_prompt,    # 10K+ tokens
        "cache_control": {"type": "ephemeral"}  # ← Cache this!
    }],
    messages=[{"role": "user", "content": "Quick question about it"}]
)
# First call: normal price. Subsequent calls: 90% cheaper for cached portion!

# 3. Semantic cache (for similar-but-not-identical queries)
# Embed the query, check if a similar query exists in cache
def semantic_cache_lookup(query: str, threshold: float = 0.95) -> str | None:
    query_emb = embed(query)
    results = cache_collection.query(query_embeddings=[query_emb], n_results=1)
    if results["distances"][0][0] < (1 - threshold):
        return results["documents"][0][0]  # Cache hit!
    return None
```

> **SPEAKER NOTES:**
> "Three levels of caching. Exact match: hash the input, check Redis. Simple but only catches identical queries. Anthropic's prompt caching is brilliant — if your system prompt is long (like RAG context), you cache it server-side and pay 90% less on subsequent calls with the same prefix. Semantic caching uses embeddings to find similar-enough past queries. If someone asked 'what's our PTO policy' yesterday and someone asks 'how many vacation days do I get' today, a semantic cache can serve the cached answer."

---

## Slide 5: Batching — 50% Off

```python
# OpenAI Batch API — for non-real-time workloads
import json

# 1. Create a JSONL file of requests
requests = [
    {
        "custom_id": f"request-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": f"Summarize article {i}"}]
        }
    }
    for i in range(1000)
]

with open("batch_input.jsonl", "w") as f:
    for r in requests:
        f.write(json.dumps(r) + "\n")

# 2. Upload and submit
batch_file = client.files.create(file=open("batch_input.jsonl", "rb"), purpose="batch")
batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"  # Results within 24 hours, 50% cheaper
)

# 3. Check status and retrieve results
status = client.batches.retrieve(batch.id)
print(f"Status: {status.status}")  # validating → in_progress → completed
```

**Use cases:** Nightly report generation, bulk classification, dataset labeling, eval suites

> **SPEAKER NOTES:**
> "If you don't need real-time responses, use the batch API. You submit a file of requests, OpenAI processes them within 24 hours, and you pay 50% less. Perfect for: nightly summarization jobs, bulk document processing, running your eval suite. It's free money on the table."

---

## Slide 6: AI Gateways & Routers

```python
# LiteLLM — unified API for all providers
from litellm import completion

# Same interface, any provider
response = completion(model="gpt-5.2", messages=[...])          # OpenAI
response = completion(model="claude-sonnet-4-6-20250217", messages=[...])   # Anthropic
response = completion(model="gemini/gemini-3-pro", messages=[...])  # Google
response = completion(model="ollama/llama3", messages=[...])   # Local

# Fallback chain
response = completion(
    model="gpt-5.2",
    messages=[...],
    fallbacks=["claude-sonnet-4-6-20250217", "gemini-3-pro"],  # Auto-fallback
    num_retries=2
)
```

| Gateway | Type | Key Feature |
|---------|------|-------------|
| **LiteLLM** | OSS proxy | Unified API, 100+ models |
| **Portkey** | Cloud | Caching, fallbacks, analytics |
| **OpenRouter** | Cloud marketplace | Pay-per-use any model |
| **Martian** | Cloud | Smart model routing |

> **SPEAKER NOTES:**
> "An AI gateway sits between your code and the model providers. Why? Three reasons: unified API (same code calls any model), fallbacks (if OpenAI is down, automatically try Anthropic), and observability (one place to see all your AI usage). LiteLLM is the most popular open-source option. OpenRouter is great if you want access to tons of models without managing API keys for each. For production, a gateway is a must-have."

---

## Slide 7: Production Patterns

```python
import time
from openai import OpenAI, RateLimitError, APITimeoutError

client = OpenAI(timeout=30.0, max_retries=3)

# 1. Streaming — don't make users stare at a spinner
def stream_response(messages: list):
    stream = client.chat.completions.create(
        model="gpt-5.2", messages=messages, stream=True
    )
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            print(token, end="", flush=True)
            full_response += token
    return full_response

# 2. Retries with exponential backoff
def resilient_call(messages: list, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5.2", messages=messages
            )
            return response.choices[0].message.content
        except RateLimitError:
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except APITimeoutError:
            print(f"Timeout, attempt {attempt + 1}/{max_retries}")
    raise Exception("All retries failed")

# 3. Timeouts — don't let requests hang forever
response = client.chat.completions.create(
    model="gpt-5.2",
    messages=messages,
    timeout=15.0  # 15-second timeout
)
```

> **SPEAKER NOTES:**
> "Three patterns you need on day one. Streaming: users will wait 30 seconds for a streaming response but abandon after 5 seconds of a spinner. Always stream in user-facing apps. Retries with backoff: API calls fail — rate limits, timeouts, server errors. Exponential backoff is the standard pattern. Timeouts: set them explicitly. A hanging request is worse than a failed one. The OpenAI SDK has built-in retries and timeout support — use it."

---

## Slide 8: Error Handling for AI

```python
from openai import (
    OpenAI, APIError, RateLimitError, APITimeoutError, 
    BadRequestError, AuthenticationError
)

def safe_llm_call(messages: list) -> dict:
    """Production-grade LLM call with proper error handling."""
    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=messages,
            timeout=30.0
        )
        return {
            "success": True,
            "content": response.choices[0].message.content,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        }
    except AuthenticationError:
        return {"success": False, "error": "Invalid API key", "retry": False}
    except BadRequestError as e:
        # Usually: context too long, invalid model, bad parameters
        return {"success": False, "error": f"Bad request: {e}", "retry": False}
    except RateLimitError:
        return {"success": False, "error": "Rate limited", "retry": True}
    except APITimeoutError:
        return {"success": False, "error": "Timeout", "retry": True}
    except APIError as e:
        return {"success": False, "error": f"API error: {e}", "retry": True}
```

> **SPEAKER NOTES:**
> "Handle each error type differently. Auth errors: don't retry, fix your key. Bad request: don't retry, fix your input (usually context too long). Rate limits and timeouts: retry with backoff. Server errors: retry a few times then fail gracefully. Always return structured results so your app can handle failures gracefully instead of crashing."

---

## Slide 9: Coding Tools — The New IDE

| Tool | Type | Best For | Price |
|------|------|----------|-------|
| **GitHub Copilot** | Autocomplete + chat | In-editor completion | $10-39/mo |
| **Cursor** | AI-native IDE | Full codebase understanding | $20/mo |
| **Claude Code** | CLI agent | Terminal-first, agentic coding | Usage-based |
| **Windsurf** | AI IDE | Similar to Cursor | $15/mo |
| **Aider** | CLI tool | Git-integrated AI coding | Free (OSS) |

> **SPEAKER NOTES:**
> "Quick tour of AI coding tools since you're all developers. Copilot is great for autocomplete — it predicts your next line. Cursor is a full IDE that understands your entire codebase and can make multi-file changes. Claude Code is a terminal agent that can plan, code, test, and commit. Aider is the open-source CLI option. My honest take: use Copilot for day-to-day autocomplete, Cursor or Claude Code for bigger tasks. Try them all — they have different strengths."

---

## Slide 10: Privacy & Enterprise Concerns

**The questions your security team will ask:**

| Concern | Question | Answer |
|---------|----------|--------|
| **Data retention** | Do they train on our data? | OpenAI API: no. Consumer: yes. |
| **Data residency** | Where is data processed? | Check provider's regions |
| **SOC 2** | Are they compliant? | OpenAI ✅ Anthropic ✅ Google ✅ |
| **BAA** | HIPAA support? | Available on enterprise tiers |
| **Self-hosting** | Can we run it ourselves? | Open source models via Ollama/vLLM |

**The honest matrix:**
```
                Privacy ◄──────────────────────► Capability
Self-hosted     ████████████                     ████░░░░░░░░
(Llama 3)

Azure OpenAI    ████████░░░░                     ████████████
(Your tenant)

API Direct      ████░░░░░░░░                     ████████████
(OpenAI/Anthropic)
```

> **SPEAKER NOTES:**
> "If you're at a company, your security team will have questions. Key facts: API usage is NOT used for training by any major provider — that's the consumer products. All major providers have SOC 2. For maximum privacy, self-host open source models. The middle ground: Azure OpenAI or AWS Bedrock gives you GPT-5.2 or Claude in your own cloud tenant with enterprise agreements. Know these answers before your security review."

---

## Slide 11: OpenClaw — What Is It?

**OpenClaw** = An open-source AI assistant platform

**Key concepts:**
- **Agent** — A running OpenClaw instance with personality + capabilities
- **Skills** — Plugins that give the agent new abilities
- **ClawHub** — Marketplace to share and discover skills
- **MCP Integration** — Skills can expose or consume MCP servers
- **Multi-channel** — Discord, Slack, Telegram, CLI, web

```
┌──────────────────────────────────────────┐
│              Your OpenClaw Agent          │
├──────────┬───────────┬───────────────────┤
│  Skill A │  Skill B  │     Skill C       │
│ (weather)│ (code run)│ (your custom!)    │
├──────────┴───────────┴───────────────────┤
│           Core Engine (LLM + Tools)       │
├──────────────────────────────────────────┤
│     Channels: Discord / Slack / CLI      │
└──────────────────────────────────────────┘
```

> **SPEAKER NOTES:**
> "OpenClaw is an open-source platform for building AI assistants. Think of it as the infrastructure layer — it handles the LLM calls, tool routing, channels (Discord, Slack, etc.), and persistence. You extend it with Skills — modular plugins that give your agent new capabilities. ClawHub is where you share skills with the community. Today we're building a skill from scratch."

---

## Slide 12: OpenClaw Skill Anatomy

```
my-skill/
├── skill.json          # Skill manifest (name, description, config)
├── index.js            # Entry point (or index.py for Python)
├── README.md           # Documentation
└── package.json        # Dependencies (Node.js skills)
```

**skill.json — The manifest:**
```json
{
  "name": "weather-lookup",
  "version": "1.0.0",
  "description": "Get current weather for any city",
  "author": "yourname",
  "main": "index.js",
  "tools": [
    {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string",
            "description": "City name"
          }
        },
        "required": ["city"]
      }
    }
  ],
  "config": {
    "WEATHER_API_KEY": {
      "description": "API key for weather service",
      "required": true,
      "secret": true
    }
  }
}
```

> **SPEAKER NOTES:**
> "A skill has four parts. The manifest (skill.json) declares what the skill does, what tools it provides, and what configuration it needs. The entry point implements the tool handlers. The README documents how to use it. And package.json lists dependencies. The tools array is the key part — this tells OpenClaw what functions to expose to the LLM, just like OpenAI function calling."

---

## Slide 13: Building a Skill — The Code

```javascript
// index.js — Skill implementation
const axios = require('axios');

module.exports = {
  // Called when the skill is loaded
  async onLoad(context) {
    console.log('Weather skill loaded!');
    // context.config has your WEATHER_API_KEY
  },

  // Tool handlers — one per tool defined in skill.json
  tools: {
    async get_weather({ city }, context) {
      const apiKey = context.config.WEATHER_API_KEY;
      
      try {
        const response = await axios.get(
          `https://api.openweathermap.org/data/2.5/weather`,
          { params: { q: city, appid: apiKey, units: 'metric' } }
        );
        
        const data = response.data;
        return {
          city: data.name,
          temperature: `${data.main.temp}°C`,
          condition: data.weather[0].description,
          humidity: `${data.main.humidity}%`,
          wind: `${data.wind.speed} m/s`
        };
      } catch (error) {
        return { error: `Could not fetch weather for ${city}: ${error.message}` };
      }
    }
  }
};
```

> **SPEAKER NOTES:**
> "The implementation is straightforward. Export an object with an optional onLoad hook and a tools object. Each tool handler receives the parameters (matching your skill.json schema) and a context object with your config. Return a result and OpenClaw handles the rest — formatting it for the LLM, rendering it in the chat. Error handling is important — return a useful error message, don't throw."

---

## Slide 14: Python Skills

```python
# index.py — Python skill implementation
import requests

class Skill:
    def __init__(self, config):
        self.api_key = config.get("WEATHER_API_KEY")
    
    def get_weather(self, params, context):
        """Tool handler for get_weather."""
        city = params["city"]
        
        try:
            response = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": city, "appid": self.api_key, "units": "metric"}
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                "city": data["name"],
                "temperature": f"{data['main']['temp']}°C",
                "condition": data["weather"][0]["description"],
                "humidity": f"{data['main']['humidity']}%"
            }
        except Exception as e:
            return {"error": f"Weather lookup failed: {str(e)}"}
```

> **SPEAKER NOTES:**
> "If you prefer Python, OpenClaw supports Python skills too. Same structure — a class with tool handler methods. The method names match the tool names in your skill.json. OpenClaw spins up a Python process and communicates via stdin/stdout. Use whichever language you're comfortable with."

---

## Slide 15: Publishing to ClawHub

```bash
# 1. Test locally
openclaw skill test ./my-skill

# 2. Validate
openclaw skill validate ./my-skill

# 3. Publish
openclaw skill publish ./my-skill

# Your skill is now available at:
# https://clawhub.com/yourname/weather-lookup
```

**ClawHub listing includes:**
- README rendered as documentation
- Install count and ratings
- Version history
- Configuration requirements
- Compatibility info

> **SPEAKER NOTES:**
> "Publishing is three commands. Test locally to make sure it works. Validate checks your skill.json, required files, and schema. Publish pushes it to ClawHub. Anyone running OpenClaw can then install your skill. Think of it like npm publish or publishing a VS Code extension."

---

## Slide 16: The AI Engineer Toolkit Map

```
┌─────────────────────────────────────────────────────────────┐
│                   THE AI ENGINEER TOOLKIT                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  MODELS          APIs              FRAMEWORKS                │
│  ├── GPT-5.2      ├── OpenAI        ├── LangGraph             │
│  ├── Claude      ├── Anthropic     ├── CrewAI                │
│  ├── Gemini      ├── Google        ├── Pydantic AI           │
│  ├── Llama       └── Ollama        └── OpenAI Agents SDK     │
│  └── Mistral                                                 │
│                                                              │
│  DATA            OBSERVABILITY     SECURITY                  │
│  ├── RAG         ├── Langfuse      ├── Input validation      │
│  ├── Chroma      ├── LangSmith     ├── Output guardrails     │
│  ├── pgvector    ├── Evals         ├── Injection defense     │
│  └── Embeddings  └── LLM-as-judge  └── PII detection         │
│                                                              │
│  PRODUCTION      TOOLS             PLATFORMS                 │
│  ├── LiteLLM     ├── Cursor        ├── OpenClaw              │
│  ├── Caching     ├── Claude Code   ├── MCP                   │
│  ├── Batching    ├── Copilot       └── ClawHub               │
│  └── Streaming   └── Aider                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

> **SPEAKER NOTES:**
> "Here's everything we've covered in 8 sessions, on one slide. You don't need to master all of this. Pick your path based on what you're building. Building a chatbot? Models + APIs + basic prompting. Building an enterprise product? Add RAG, observability, and security. Building an AI platform? Add frameworks, production patterns, and tooling. You now have the map — explore based on your needs."

---

## Slide 17: What to Do Next

1. **Pick a project** — Best way to learn is to build
2. **Start simple** — API call → tool use → RAG → agent (in that order)
3. **Add observability from day one** — You'll thank yourself
4. **Join communities** — OpenClaw Discord, LangChain Discord, r/LocalLLaMA
5. **Stay current** — This field changes weekly (not monthly)

**Resources:**
- OpenAI Cookbook: github.com/openai/openai-cookbook
- Anthropic docs: docs.anthropic.com
- LangGraph tutorials: langchain-ai.github.io/langgraph
- OpenClaw: github.com/openclaw
- This course materials: [your repo]

> **SPEAKER NOTES:**
> "My final advice: build something. Pick a problem at your work or a side project and apply what you've learned. Start with the simplest approach that could work — usually a good prompt with the right model. Only add complexity (RAG, agents, multi-model) when you have a reason. Ship it, get feedback, iterate. You now know more about AI engineering than 95% of developers. Go build."

---

## Slide 18: Hands-On — Build an OpenClaw Skill

We're building a **code snippet manager** skill:
- Save code snippets with tags
- Search snippets by keyword or tag
- List recent snippets

**Open:** `session-8/code/openclaw_skill/`

> **SPEAKER NOTES:**
> "For our final hands-on, we're building a practical OpenClaw skill — a code snippet manager. You'll be able to save, search, and list code snippets through your AI assistant. This exercises everything: skill structure, tool definitions, data persistence, and error handling. Let's build it."

---

## Slide 19: Recap & Course Wrap-Up

**8 sessions, you now know:**
1. ✅ The AI landscape — models, providers, APIs
2. ✅ Prompting, structured output, multimodal
3. ✅ Tool use & custom assistants
4. ✅ MCP & plugin ecosystems
5. ✅ Agentic AI & frameworks
6. ✅ RAG & data pipelines
7. ✅ Observability, evals & security
8. ✅ Production patterns, dev tools & platforms

**You are now an AI engineer. Go build. 🚀**

> **SPEAKER NOTES:**
> "That's a wrap on Cut the Crap — AI Engineer Edition. In 16 hours we went from API calls to production-grade AI systems. You understand the landscape, can build with any provider, know how to add your own data, make agents that reason, monitor and test everything, and deploy to production. The field moves fast but the fundamentals don't change — good prompts, good data, good engineering. Thanks for being here. Now go build something amazing."
