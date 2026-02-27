# Part 6: OpenClaw & Setup
## Topics 14–17 | ~50 minutes

---

## Slide 14: Meet OpenClaw — Live Demo

**SHOW:**
- OpenClaw running on phone (Discord or native)
- Title: "One app. Every AI brain. Pay only for what you use."

**SPEAKER NOTES:**
> OK, remember the Netflix analogy? ChatGPT is Netflix — one company's content. What if you had an app that combined Netflix, Disney+, HBO, and Prime into one interface?
>
> That's OpenClaw.
>
> OpenClaw is a personal AI assistant that can use ANY model — GPT-4o, Claude Sonnet, Gemini, whatever — through one interface. It runs on your phone, your computer, or in Discord. And instead of paying $20/month to each company, you pay pennies per message through API keys.

### Live Demo on Phone

**Show on screen (phone mirrored or screenshots):**

1. Open Discord on phone → Show OpenClaw channel
2. Type a message: `What's the weather going to be like in Toronto this weekend?`
3. Show the response
4. Say: `/model claude-sonnet` → Type the same question → Show different response
5. Say: `/model gpt-4o` → Same question → Third response
6. Point out: "Same app, three different brains. I just switched like changing a TV channel."

**SPEAKER NOTES (continued):**
> Watch what just happened. I asked the same question three times, but each time I picked a different AI brain. I didn't need three apps, three accounts, three subscriptions. One tool, all brains.
>
> And look at the cost — [show cost if visible] — that conversation cost me about 2 cents. Not $20/month. Two cents.

---

## Slide 15: Switch Models, Show Cost

**SHOW:**

### Why This Matters — Real Cost Comparison

| Scenario | ChatGPT Plus | Claude Pro | OpenClaw (API) |
|----------|-------------|-----------|----------------|
| 10 messages/day, casual use | $20/mo | $20/mo | ~$3-5/mo |
| Heavy use + all models | $20/mo (one company) | $20/mo (one company) | ~$10-20/mo (ALL companies) |
| Light use (few times/week) | $20/mo (overpaying) | $20/mo (overpaying) | ~$1-2/mo |

**SPEAKER NOTES:**
> Let me show you the model-switching in action with a real workflow.

### Live Demo: Smart Model Switching

**Step 1 — Quick question, use the cheap model:**
```
/model claude-haiku
What's a good substitute for buttermilk in baking?
```
> Cost: ~$0.0005. Basically free. Haiku handles this perfectly.

**Step 2 — Real work, use the balanced model:**
```
/model claude-sonnet
Review this email and make it more professional but keep my personality:

hey sarah, so the project is gonna be like 2 weeks late because our supplier messed up. sorry about that. we're on it tho and will keep you posted. thanks for being cool about it
```
> Cost: ~$0.01. Still cheap. Sonnet does a great job here.

**Step 3 — Complex analysis, bring out the big guns:**
```
/model claude-opus
I'm considering moving from Toronto to Calgary. Analyze the financial implications: housing costs, taxes (provincial + federal), cost of living, job market for marketing professionals. Give me a clear recommendation with numbers.
```
> Cost: ~$0.05. More expensive but worth it for this complexity.

> See the pattern? Match the brain to the task. You wouldn't take a taxi to the mailbox, and you wouldn't bike to the airport. Same idea.

---

## Slide 16: Setup Guide

**SHOW:**
- Title: "Let's get you set up. 30 minutes. You'll leave here with a working setup."

### Step 1: Get Your API Keys (~10 min)

**SHOW on screen, walk through together:**

#### OpenAI API Key (for GPT models)
1. Go to **platform.openai.com**
2. Sign up or log in (separate from your ChatGPT account!)
3. Click **"API keys"** in the left sidebar
4. Click **"Create new secret key"**
5. Name it: `openclaw`
6. **COPY IT NOW** — you can't see it again
7. Paste it somewhere safe (Notes app, password manager)
8. Go to **Billing** → Add $10 credit (this will last you weeks)

#### Anthropic API Key (for Claude models)
1. Go to **console.anthropic.com**
2. Sign up or log in
3. Click **"API Keys"**
4. Click **"Create Key"**
5. Name it: `openclaw`
6. **COPY IT NOW**
7. Go to **Billing** → Add $10 credit

#### Google AI API Key (for Gemini models)
1. Go to **aistudio.google.com**
2. Sign in with your Google account
3. Click **"Get API key"** → **"Create API key"**
4. **COPY IT**
5. Gemini has a generous free tier — you may not need to add credit yet

**SPEAKER NOTES:**
> I know this feels like a lot of copying and pasting. It is. You do it once and you're done forever. Think of it as setting up your Wi-Fi — annoying for 10 minutes, then you never think about it again.
>
> If you get stuck, raise your hand. The most common issue is people confusing their ChatGPT login with the API platform login — they're separate accounts/sites.

### Step 2: Install OpenClaw (~15 min)

**SHOW: Two paths based on operating system**

---

#### Windows Users (WSL Required)

**SPEAKER NOTES:**
> Windows users, we need to install WSL first. WSL stands for "Windows Subsystem for Linux" — think of it as a little Linux computer inside your Windows computer. OpenClaw runs inside it. This is the most annoying part. Once it's done, everything else is easy.

**Step-by-step (type/paste these commands):**

**1. Install WSL** (open PowerShell as Administrator):
```powershell
wsl --install
```
> Restart your computer when prompted. Yes, really. Then come back here.

**2. After restart, open Ubuntu** (it'll appear in your Start menu):
- It will ask you to create a username and password
- Pick something simple you'll remember
- **The password won't show dots as you type — that's normal, just type and press Enter**

**3. Update everything** (in Ubuntu terminal):
```bash
sudo apt update && sudo apt upgrade -y
```

**4. Install Node.js:**
```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
```

**5. Install OpenClaw:**
```bash
npm install -g @anthropic/openclaw
```

**6. Set up OpenClaw:**
```bash
openclaw setup
```
> This will walk you through pasting in your API keys from Step 1.

---

#### macOS Users

**Step-by-step:**

**1. Open Terminal** (press Cmd+Space, type "Terminal", hit Enter)

**2. Install Homebrew** (if you don't have it):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**3. Install Node.js:**
```bash
brew install node
```

**4. Install OpenClaw:**
```bash
npm install -g @anthropic/openclaw
```

**5. Set up OpenClaw:**
```bash
openclaw setup
```
> Same as Windows — paste in your API keys when prompted.

---

### Step 3: Verify It Works (~5 min)

**In your terminal (both Windows WSL and macOS):**
```bash
openclaw chat "Hello! Tell me a fun fact about the city I'm in."
```

> If you see a response, you're done. You now have every major AI brain available from your command line.

**SPEAKER NOTES:**
> Hands up if you got a response! [celebrate]
>
> If it didn't work, don't panic. The most common issues:
> - "command not found" → Node didn't install correctly. Run `node --version` to check.
> - "invalid API key" → You copied it wrong. Go back to the platform site and create a new one.
> - "insufficient funds" → You need to add credit to your API account.
>
> I'll stay after the session to help anyone who's stuck.

---

## Slide 17: Cheat Sheet + Q&A

**SHOW:**
- "Grab your cheat sheet" — point to handout or URL
- Title: "What to remember from today"

**SPEAKER NOTES:**
> Here's your cheat sheet [hand out or share link]. It's got:
> - The three AIs and when to use each one
> - Model names (fast vs smart) for each company
> - How much things actually cost
> - The privacy rules
> - Your setup commands for reference
>
> **The one thing I want you to take away:** You walked in knowing ChatGPT. You're walking out knowing how to use ANY AI, switch between them, and pay a fraction of what a subscription costs. That's real power.
>
> Questions? Let's do it. Nothing is too basic — if you're wondering it, three other people are too.

### Common Q&A Topics (prep these):

**"Is AI going to take my job?"**
> Not directly. But someone who uses AI well might outperform someone who doesn't. The goal is to be the person who uses it well.

**"Which AI should I use day-to-day?"**
> Start with Claude Sonnet for most things — it's the best all-rounder right now. Use GPT-4o when you need image generation. Use Gemini when you need current information or Google integration.

**"Is my data safe?"**
> API usage is the safest option. Don't put passwords or truly confidential info in any AI. Treat it like a smart coworker.

**"What about open source AI? What about Llama?"**
> Great question. There are open-source models you can run on your own computer. They're getting good but still behind the big three for most tasks. That's a topic for an advanced session.

**"How do I keep up with all this? It changes so fast."**
> Follow a couple of good sources: The Verge AI section, Simon Willison's blog, and just... use the tools regularly. You learn by doing.
