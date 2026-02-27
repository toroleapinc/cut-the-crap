# Part 2: How It Works
## Topics 4–7 | ~30 minutes

---

## Slide 4: Models — Fast/Cheap vs Smart/Expensive

**SHOW:**

| Tier | Examples | Speed | Cost | Good For |
|------|----------|-------|------|----------|
| 🐇 **Fast & Cheap** | GPT-4o-mini, Claude Haiku, Gemini Flash | ~2 sec | ~$0.001/msg | Quick questions, simple tasks, brainstorming |
| 🦊 **Balanced** | GPT-4o, Claude Sonnet, Gemini Pro | ~5 sec | ~$0.01/msg | Most real work, writing, analysis |
| 🧠 **Max Power** | o3, Claude Opus | ~15 sec | ~$0.05/msg | Complex reasoning, hard math, deep analysis |

**SPEAKER NOTES:**
> Remember the brain vs app thing? Well, brains come in different sizes.
>
> Think of it like cars:
> - **Haiku/Flash/Mini** = Honda Civic. Gets you there, cheap on gas, perfect for errands.
> - **Sonnet/GPT-4o/Pro** = BMW. Nicer ride, costs more, what you want for a road trip.
> - **Opus/o3** = Formula 1 car. Insanely powerful, expensive to run, and you only need it for the race.
>
> Here's the thing most people get wrong: **the expensive model isn't always better.** If you ask "What's the capital of France?" the Honda Civic and the F1 car give the exact same answer. The F1 car just costs 50x more to do it.
>
> The skill is matching the task to the model. Quick question? Use the fast one. Writing a business proposal? Use the balanced one. Analyzing a 50-page legal contract? Break out the big guns.

---

## Slide 5: Free Tiers vs Paid — The Pricing Truth

**SHOW:**

### What You Get for Free
| Service | Free Tier | What You Hit |
|---------|-----------|-------------|
| ChatGPT | GPT-4o-mini unlimited, GPT-4o limited | ~10-15 GPT-4o messages, then throttled |
| Claude.ai | Claude Sonnet, limited | ~20-30 messages, then locked for hours |
| Gemini | Gemini Pro, generous | Rarely hit limits for normal use |

### Paid Plans
| Service | Price | What You Get |
|---------|-------|-------------|
| ChatGPT Plus | $20/mo | More GPT-4o, o1 access, image gen, GPTs |
| Claude Pro | $20/mo | 5x more Sonnet, Opus access |
| Gemini Advanced | $20/mo | Gemini Ultra, longer context, Google integration |

### The API Route (what OpenClaw uses)
| Usage Level | Approximate Cost |
|-------------|-----------------|
| Light (few msgs/day) | $1–5/month |
| Moderate (daily use) | $5–15/month |
| Heavy (power user) | $15–40/month |

**SPEAKER NOTES:**
> Let's talk money. Because "free" in AI is like "free" at a casino — they're happy for you to walk in, but there's a catch.
>
> **Free tiers** are great for trying things out. But you'll hit walls. ChatGPT cuts you off from the good model after a handful of messages. Claude locks you out for hours. It's like a free trial that resets every day.
>
> **$20/month plans** are the obvious next step. They're fine. But here's what bugs me: you're paying $20 to ONE company. What if Claude is better for your writing but ChatGPT is better for your images? Now you're paying $40/month for two subscriptions.
>
> **The API route** is the third option — and it's what OpenClaw uses. Instead of a flat $20/month, you pay per message. Like a pay-as-you-go phone instead of a plan. For most people, that's $5-15/month AND you get access to ALL the brains, not just one company's. We'll set this up in Part 6.

---

## Slide 6: What's an API Key?

**SHOW:**
```
Your API Key:
sk-proj-abc123def456ghi789...

DON'T SHARE THIS. It's like a credit card number.
```

**SPEAKER NOTES:**
> OK, "API key." Sounds technical. Let me make it simple.
>
> You know those **prepaid gas cards**? You go to Shell, buy a $50 card, and every time you pump gas, it deducts from the card. You don't need to sign a contract. You don't need a monthly plan. You just load money, use gas, done.
>
> An API key is your prepaid gas card for AI.
>
> Here's how it works:
> 1. You go to OpenAI's website (or Anthropic's, or Google's)
> 2. You create an account and add $10 to your balance (like loading a gift card)
> 3. They give you a long code — that's your API key
> 4. You paste that code into a tool like OpenClaw
> 5. Every time you send a message, it deducts a tiny amount from your balance
>
> That's it. The key is just proof that you've paid. It's your gas card number.
>
> **Important:** Treat it like a credit card number. If someone gets your API key, they can use your balance. Don't post it online, don't share it in a screenshot. We'll set one up together in Part 6.

**SHOW (diagram):**
```
You → load $10 → OpenAI gives you: sk-proj-abc123...
                                         ↓
               Paste into OpenClaw → sends messages → deducts pennies
                                         ↓
                              Balance: $10.00 → $9.98 → $9.95 ...
```

---

## Slide 7: OAuth vs API Key — Two Ways to Log In

**SHOW:**

| | OAuth ("Sign in with...") | API Key |
|---|---|---|
| **Analogy** | Hotel key card — front desk verifies you, gives you a card | Prepaid gas card — you loaded money, use anytime |
| **How it feels** | Click "Sign in with Google" → done | Copy-paste a long code into settings |
| **Who tracks cost** | The app (included in subscription) | You (pay-per-use from your balance) |
| **Example** | Using ChatGPT's website | Using OpenClaw with your own key |
| **Flexibility** | Stuck with that app's options | Use any tool that accepts the key |

**SPEAKER NOTES:**
> Quick concept, then we move on.
>
> When you use ChatGPT's website, you log in with your email and password (or "Sign in with Google"). That's **OAuth**. It's like checking into a hotel — the front desk verifies who you are and gives you a key card. Simple. But you can only use it at THAT hotel.
>
> An **API key** is different. It's like that prepaid gas card — it works at any pump that accepts it, not just one gas station. You can use your OpenAI API key in ChatGPT's app, OR in OpenClaw, OR in dozens of other tools. Freedom.
>
> For today: OAuth = simple login on one website. API key = portable pass you can use anywhere. You already use OAuth every day. We'll add API keys in Part 6. Not scary, I promise.
>
> You don't need to remember the technical terms. Just remember: **website login** vs **portable code**. That's it. Let's move on.
