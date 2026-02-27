# Cut the Crap — AI Engineer Edition
## Developer: 8 × 2-Hour Series

### Updated: February 2026
### Current Models & SDKs
- **OpenAI**: GPT-5.2 (flagship reasoning), GPT-5 (reasoning), GPT-5-mini, GPT-4.1 (best non-reasoning), GPT-4.1-mini, GPT-4.1-nano
- **Anthropic**: Claude Opus 4.6, Claude Sonnet 4.6, Claude Haiku 3.5
- **Google**: Gemini 2.5 Pro, Gemini 2.5 Flash
- **Image Gen**: GPT Image 1.5, Sora 2 (video)
- **Audio**: gpt-audio-1.5, gpt-realtime-1.5, Whisper
- **Embeddings**: text-embedding-3-small, text-embedding-3-large
- **Coding Agents**: GPT-5.2-Codex, Claude Code, Cursor, GitHub Copilot

### Session 1: The AI Landscape & APIs
1. Landscape: OpenAI, Anthropic, Google, Meta, Mistral, DeepSeek
2. Models deep dive: types, reasoning (GPT-5/o3/o4-mini) vs non-reasoning (GPT-4.1), temperature, structured output
3. Open source models: Llama, Mistral, DeepSeek, Qwen
4. Self-hosting: Ollama, vLLM
5. API key setup — live first call (OpenAI + Anthropic)
6. OAuth vs API key
7. Hands-on: build a simple chat script (Python) — `chat_script.py`

### Session 2: Prompting, Structured Output & Multimodal
8. Prompt engineering: system prompts, few-shot, chain-of-thought
9. Structured output & JSON schemas (OpenAI native, Anthropic tool-use, Gemini response_schema)
10. Multimodal APIs: vision (GPT-4.1, Claude Sonnet 4.6, Gemini 2.5), audio (Whisper/gpt-audio-1.5), image gen (GPT Image 1.5), video (Sora 2)
11. Hands-on: build a multimodal app — `multimodal_app.py`

### Session 3: Tool Use & Custom Assistants
12. Function calling / tool use (all 3 providers)
13. Assistants API (OpenAI) vs Messages API (Anthropic)
14. Custom GPTs vs Skills vs Gems — build one each
15. Hands-on: build a tool-calling assistant — `tool_calling.py`

### Session 4: MCP, Plugins & Marketplaces
16. MCP — the universal tool protocol
17. MCP servers: filesystem, GitHub, Slack, DBs
18. Setting up MCP servers (live)
19. Marketplaces: GPT Store, ClawHub, community MCP
20. Hands-on: connect MCP to Claude Desktop / OpenClaw

### Session 5: Agentic AI & Frameworks
21. What is an agent? (ReAct loop)
22. Frameworks: LangChain/LangGraph, CrewAI, OpenAI Agents SDK, Anthropic tool use, AutoGen, Pydantic AI
23. Multi-agent orchestration
24. Agent memory: short-term vs long-term
25. Agentic coding: Codex (GPT-5.2-Codex), Claude Code, Cursor
26. Hands-on: build a multi-step agent with LangGraph — `langgraph_agent.py`

### Session 6: RAG & Data
26. RAG: embeddings (text-embedding-3-small/large) → vector DB → retrieval → generation
27. Tools: Pinecone, ChromaDB, pgvector, Weaviate
28. When RAG vs long context window (200k+ tokens) vs fine-tuning
29. Fine-tuning: OpenAI fine-tuning (GPT-4.1), LoRA
30. Context window management strategies
31. Hands-on: build a RAG pipeline — `rag_pipeline.py`

### Session 7: Observability, Evals & Security
32. Tracing & observability: LangSmith, Langfuse (decorator-based @observe), Braintrust, Arize
33. Live: trace an agent run with Langfuse
34. Evals: testing non-deterministic outputs, LLM-as-judge (GPT-4.1)
35. Hallucinations & limitations
36. Prompt injection & security (pattern matching + LLM classifier)
37. Guardrails: input/output validation, PII, toxicity
38. Hands-on: add tracing + evals to Session 5 agent — `tracing_demo.py`

### Session 8: Production, Dev Tools & OpenClaw
39. Cost optimization: model routing (GPT-4.1-nano → GPT-4.1 → GPT-5), caching, batching
40. AI gateways/routers: LiteLLM, Portkey, OpenRouter
41. Production: rate limits, retries, streaming, error handling
42. Coding tools: Claude Code vs Cursor vs Copilot vs Codex
43. Privacy & data concerns for enterprise
44. OpenClaw deep dive: skills, agents, ClawHub
45. Hands-on: build your own OpenClaw skill — `openclaw_skill/`
46. Full picture: the AI engineer toolkit map
47. Q&A
