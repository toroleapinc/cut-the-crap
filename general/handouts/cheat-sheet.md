# Cut the Crap — AI Cheat Sheet (Feb 2026)

## The Big Three AIs

| AI | Made By | Best At | Sign Up |
|----|---------|---------|---------|
| **ChatGPT** | OpenAI | Image gen, all-rounder, plugins, coding (GPT-5.2-Codex) | [chatgpt.com](https://chatgpt.com) |
| **Claude** | Anthropic | Writing, analysis, long docs (1M tokens), careful answers | [claude.ai](https://claude.ai) |
| **Gemini** | Google | Google integration, Deep Research, generous free tier | [gemini.google.com](https://gemini.google.com) |

## Models: Pick the Right Brain

| Need | OpenAI | Anthropic | Google |
|------|--------|-----------|--------|
| Quick & cheap | GPT-4.1-nano / GPT-5-mini | Haiku 3.5 | Gemini 2.5 Flash |
| Everyday work | GPT-4.1 / GPT-5.2 | **Sonnet 4.6** ⭐ | Gemini 2.5 Pro |
| Maximum power | GPT-5.2-Codex | **Opus 4.6** | Gemini 2.5 Pro (Thinking) |

⭐ **Sonnet 4.6** is now the default for free Claude users — near-Opus quality at a fraction of the cost.

**Rule of thumb:** Start with the mid-tier model. Only go up for deep analysis or complex coding.

## What Things Actually Cost

| Tier | ChatGPT | Claude | Gemini |
|------|---------|--------|--------|
| **Free** | GPT-5.2 (limited, has ads) | Sonnet 4.6 (limited) | 2.5 Flash + limited 2.5 Pro |
| **Budget** | Go — $8/mo | — | AI Plus — $8/mo |
| **Standard** | Plus — $20/mo | Pro — $20/mo | AI Pro — $20/mo |
| **Power** | Pro — $200/mo | Max 5x — $100/mo | AI Ultra — $250/mo |
| **OpenClaw** (OAuth or API) | **OAuth: $0 extra (uses your sub) · API: $5–15/mo typical** |||

💡 **Best value path:** Use free tiers to learn → pick ONE $20 sub → graduate to OpenClaw + API for max flexibility.

## Top 3 Prompting Tips

1. **Be specific.** Not "write an email" → "Write a 100-word professional email to Sarah apologizing for a 2-week delay. Tone: warm but not groveling."
2. **Give context + constraints.** State your role, audience, format, word count, and tone.
3. **Iterate.** First answer not great? Say "Make it more concise" or "Focus more on X." AI is a conversation, not a slot machine.

## Hallucination Check — 30-Second Rule

Before trusting AI on **facts**:
1. 🔍 Google the specific claim
2. 🤖 Ask a second AI the same question
3. ❓ Ask: "Are you sure? Can you provide a source?"
4. 📚 For anything important: verify with a real source

**AI is great at thinking, mediocre at remembering.**

## Two Ways to Authenticate: OAuth vs API Key

### OAuth (Use Your Existing Subscription)
- Available in **official apps only**: OpenClaw, Claude Code, Codex CLI
- The app gives you a link → you authorize in browser → paste code back
- Uses your existing Pro/Max plan — **no extra cost**, but tokens count against subscription limits
- You see token usage only (no dollar cost shown)
- Credentials stored locally:
  - OpenClaw: `~/.openclaw/credentials/oauth.json`
  - Claude Code: `~/.claude/.credentials.json`

### API Key (Pay-as-You-Go)
- For **any app**, including your own custom programs
- Go to provider console → create key → add billing/credits
- You pay per token — typically **pennies per conversation**
- You see **exact dollar cost**
- ⚠️ **Custom programs can ONLY use API keys** — OAuth is not available for your own code

### API Key Setup (One-Time, ~5 min each)

| Provider | URL | What to do |
|----------|-----|-----------|
| OpenAI | [platform.openai.com](https://platform.openai.com) | API Keys → Create → Add $10 billing |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | API Keys → Create → Add $10 billing |
| Google | [aistudio.google.com](https://aistudio.google.com) | Get API Key → Create (free tier very generous) |

**Treat API keys like credit card numbers. Never share them.**

## OpenClaw Quick Install

During `openclaw setup`, you choose your auth method:
- **OAuth**: Anthropic (Claude Code credentials) or OpenAI (Codex) — uses your existing subscription
- **API Key**: Anthropic, OpenAI, Google, xAI, and many others — pay-as-you-go

**macOS:**
```bash
brew install node
npm i -g openclaw
openclaw setup
```

**Windows (PowerShell as Admin first):**
```powershell
wsl --install
```
Restart, open Ubuntu, then:
```bash
sudo apt update && sudo apt upgrade -y
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
npm i -g openclaw
openclaw setup
```

## What to Try First

1. **Right now (free):** Go to [claude.ai](https://claude.ai) → upload a document → ask it to summarize and find issues
2. **This week:** Try the same task on all 3 AIs — you'll quickly find your favorite
3. **When ready:** Set up OpenClaw → use any model from one place → pay only for what you use

## Keep Learning

- **Simon Willison's blog** — [simonwillison.net](https://simonwillison.net) (best AI commentary)
- **The Verge AI** — [theverge.com/ai](https://theverge.com/ai)
- **Just use it.** 10 minutes a day → fluent in a month

---
*Cut the Crap — Everything You Need to Know About AI • Handout v2.0 • February 2026*
