# Part 6: OpenClaw & Setup
## Topics 14–19 | ~50 minutes

---

## Slide 14: What is Markdown?

- Plain text with simple formatting: `# heading`, `**bold**`, `- bullet`
- No special software needed — any text editor works
- AI tools speak Markdown natively (prompts, responses, configs)
- GitHub, Discord, Slack, Notion all use Markdown
- OpenClaw config files, memory, and agents all use Markdown

### Quick Reference
| You Type | You Get | Use Case |
|----------|---------|----------|
| `# Heading` | Large heading | Sections |
| `**bold**` | **bold** | Emphasis |
| `- item` | • bullet | Lists |
| `` `code` `` | `code` | Commands |

---

## Slide 15: Linux Survival Kit

OpenClaw runs in a terminal. These 5 commands cover 90%:

| Command | What It Does | Analogy |
|---------|-------------|---------|
| `cd folder` | Go into folder | Double-click a folder |
| `cd ..` | Go back one level | Back button |
| `ls` | List files | Open folder to see contents |
| `cat file.txt` | Display file | Open a document |
| `nano file.txt` | Edit file | Notepad in terminal |
| `Ctrl+C` | Stop anything | Force-quit |

---

## Slide 16: Meet OpenClaw

- One app, every AI brain: GPT-5.2, Claude Opus 4.6, Gemini 3 Pro — all through one interface
- Runs on phone (Discord), computer (CLI), or web
- Pay pennies per message via API instead of $20/mo subscriptions
- Live demo: switch models with `/model`, show cost per message

---

## Slide 17: Cost Comparison & Smart Model Switching

| Usage | ChatGPT Plus | Claude Pro | OpenClaw (API) |
|-------|-------------|-----------|----------------|
| Light (few/week) | $20/mo | $20/mo | ~$1-2/mo |
| Moderate (daily) | $20/mo (one co.) | $20/mo (one co.) | ~$5-15/mo (ALL) |
| Heavy (power user) | $40/mo (two subs) | $40/mo (two subs) | ~$15-25/mo |

### Smart Switching Demo
- 🐇 Quick Q → Haiku 4.5: $0.0005
- 🦊 Real work → Sonnet 4.6: $0.01
- 🧠 Deep analysis → Opus 4.6: $0.05

---

## Slide 18: Setup Guide

### Step 1: API Keys (10 min)
- **OpenAI:** platform.openai.com → API Keys → Create → add $10
- **Anthropic:** console.anthropic.com → API Keys → Create → add $10
- **Google:** aistudio.google.com → Get API Key (generous free tier)

### Step 2a: Windows (WSL)
```bash
wsl --install  # PowerShell as Admin, then restart
sudo apt update && sudo apt upgrade -y
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
npm install -g @anthropic/openclaw && openclaw setup
```

### Step 2b: macOS
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install node
npm install -g @anthropic/openclaw && openclaw setup
```

### Step 3: Verify
```bash
openclaw chat "Hello! Tell me a fun fact."
```

---

## Slide 19: Cheat Sheet + Q&A

**Key takeaways:**
- 3+ major AIs, each with strengths — don't limit yourself to one
- Brain ≠ App — pick the best model for each task
- AI hallucinates — always verify facts (30-second rule)
- API keys = freedom + savings vs subscriptions
- Start with Claude Sonnet 4.6 for most things (Feb 2026 best all-rounder)

**Common Qs:**
- "Will AI take my job?" → No, but someone using AI well might outperform you
- "Which AI daily?" → Claude Sonnet 4.6 most things, GPT-5.2 for images, Gemini for Google
- "Open source?" → Llama 4, DeepSeek V4 (94% cheaper) — advanced session topic
