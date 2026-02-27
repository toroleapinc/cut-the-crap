# Part 2: How It Works
## Topics 4–7 | ~30 minutes

---

## Slide 4: Models — Fast/Cheap vs Smart/Expensive

**SHOW:**

| Tier | Examples (Feb 2026) | Speed | Cost/msg | Good For |
|------|---------------------|-------|----------|----------|
| 🐇 Fast & Cheap | Haiku 4.5, Gemini 2.0 Flash, GPT-4o-mini | ~1-2s | ~$0.001 | Quick questions, brainstorming |
| 🦊 Balanced | Sonnet 4.6, GPT-5.2, Gemini 3 Pro | ~3-5s | ~$0.01 | Most real work, writing, analysis |
| 🧠 Max Power | Opus 4.6, o3 | ~10-15s | ~$0.05 | Complex reasoning, deep analysis |

**SAY:**
> Cars analogy: Haiku = Honda Civic, Sonnet = BMW, Opus = F1 car. The expensive model isn't always better. Match the task to the model.

---

## Slide 4b: The AI Model Landscape — February 2026

| Company | Flagship | Mid-Tier | Fast/Cheap | Reasoning |
|---------|----------|----------|------------|-----------|
| OpenAI | GPT-5.2 (Dec '25) — 400K ctx, $20/$60 per 1M tok | GPT-5.1, GPT-4o | GPT-4o-mini | o3, o4-mini |
| Anthropic | Opus 4.6 (Feb 5 '26) | Sonnet 4.6 (Feb 17 '26) | Haiku 4.5 (Oct '25) | — |
| Google | Gemini 3 Pro (Nov '25) — 1M ctx | — | Gemini 2.0 Flash (650ms avg!) | — |
| Others | Grok 4 (xAI) | Mistral Large | DeepSeek V4 (94% cheaper) | Llama 4 (open source) |

---

## Slide 5: Free Tiers vs Paid — The Pricing Truth

| Service | Free Tier | Paid ($20/mo) | API via OpenClaw |
|---------|-----------|---------------|------------------|
| ChatGPT | GPT-4o-mini unlimited, GPT-4o ~15 msg | More GPT-5.2, o3 access | ~$3-5/mo light use |
| Claude.ai | Sonnet 4.6 ~20-30 msg then locked | 5x more, Opus 4.6 access | ~$3-5/mo light use |
| Gemini | Gemini 3 Pro (generous) | Longer context, Google integration | Generous free / pennies |

**SAY:**
> Free = casino. $20/mo = one company. API = pay-as-you-go for ALL companies.

---

## Slide 6: What's an API Key?

Analogy: prepaid gas card for AI. Load $10, get a code (sk-proj-abc123...), paste into OpenClaw, every message deducts pennies. Treat like a credit card number.

---

## Slide 7: OAuth vs API Key

| | OAuth ("Sign in with...") | API Key |
|---|---|---|
| Analogy | Hotel key card | Prepaid gas card |
| How it feels | Click "Sign in with Google" | Copy-paste a long code |
| Who tracks cost | App (subscription) | You (pay-per-use) |
| Flexibility | Stuck with that app | Use any tool |

**SAY:**
> OAuth = hotel key card (one hotel). API key = gas card (any pump). You already use OAuth daily. We'll add API keys in Part 6.
