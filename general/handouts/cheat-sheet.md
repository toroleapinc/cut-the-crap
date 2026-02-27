# Cut the Crap — AI Cheat Sheet

## The Big Three AIs

| AI | Made By | Best At | Website |
|----|---------|---------|---------|
| **ChatGPT** | OpenAI | Image generation, all-rounder, plugins | chat.openai.com |
| **Claude** | Anthropic | Writing, analysis, long documents, careful answers | claude.ai |
| **Gemini** | Google | Current info, Google integration, research | gemini.google.com |

## Models: Pick the Right Brain

| Need | OpenAI | Anthropic | Google |
|------|--------|-----------|--------|
| Quick & cheap | GPT-4o-mini | Claude Haiku | Gemini Flash |
| Everyday work | GPT-4o | Claude Sonnet | Gemini Pro |
| Maximum power | o3 | Claude Opus | Gemini Ultra |

**Rule of thumb:** Use the cheap model for simple questions, balanced for real work, max power only when you need deep analysis.

## What Things Actually Cost

| How you use AI | Monthly cost |
|----------------|-------------|
| Free tiers (limited) | $0 |
| One subscription (ChatGPT/Claude/Gemini) | $20/mo |
| **OpenClaw + API keys (all models)** | **$5-15/mo typical** |

## The Privacy Rules

| ✅ Safe | ⚠️ Think Twice | ❌ Never |
|---------|---------------|---------|
| Public info | Client details | Passwords |
| Your own writing | Business strategies | Credit card / SIN |
| General questions | Financial info | Medical records with names |
| Anonymized scenarios | Anything under NDA | Others' private data |

**Best practice:** Use API (OpenClaw) for sensitive work — companies don't train on API data.

## Hallucination Check — 30-Second Rule

Before trusting AI on **facts**:
1. Google the specific claim
2. Ask a second AI the same question
3. Ask the AI: "Are you sure? Can you provide a source?"
4. For anything important: verify with a real source

**AI is great at thinking, mediocre at remembering.**

## API Key Setup (One-Time)

| Provider | URL | What to do |
|----------|-----|-----------|
| OpenAI | platform.openai.com | API Keys → Create → Add $10 billing |
| Anthropic | console.anthropic.com | API Keys → Create → Add $10 billing |
| Google | aistudio.google.com | Get API Key → Create (free tier generous) |

**Treat API keys like credit card numbers. Don't share them.**

## OpenClaw Quick Install

**macOS:**
```bash
brew install node
npm install -g @anthropic/openclaw
openclaw setup
```

**Windows (in PowerShell as Admin first):**
```powershell
wsl --install
```
Then restart, open Ubuntu, and run:
```bash
sudo apt update && sudo apt upgrade -y
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
npm install -g @anthropic/openclaw
openclaw setup
```

## Prompt Tips

| Instead of... | Try... |
|--------------|--------|
| "Write me an email" | "Write a professional 100-word email to a client named Sarah apologizing for a 2-week delay, tone: warm but not groveling" |
| "Help me with my resume" | "Review my resume for a senior marketing role. Tell me what's weak, what's missing, and rewrite the summary section" |
| "Explain AI" | "Explain how AI chatbots work to a 70-year-old who uses email but not much else. Use a cooking analogy" |

**Be specific. Give context. State the format you want. Set constraints (word count, tone, audience).**

## Keep Learning

- **The Verge** — AI section (thevrge.com/ai)
- **Simon Willison's blog** — simonwillison.net
- **Just use it.** 10 minutes a day → fluent in a month

---

*Cut the Crap — Everything You Need to Know About AI*
*Handout v1.0*
