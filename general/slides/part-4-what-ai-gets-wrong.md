# Part 4: What AI Gets Wrong
## Topics 9–10 | ~20 minutes

---

## Slide 9: Hallucinations — When AI Is Confidently Wrong

**SHOW:**
- Title: "AI doesn't know what it doesn't know"
- Example screenshot of AI confidently citing a fake study or fake book

**SPEAKER NOTES:**
> Time for the most important warning in this entire course.
>
> AI **makes things up**. Not occasionally. Regularly. And it does it with complete confidence, perfect grammar, and a straight face. This is called a "hallucination."
>
> It's not lying — it doesn't have intentions. It's more like a very confident student who didn't study: instead of saying "I don't know," it generates something that SOUNDS right.

### Live Demo: Catching a Hallucination

**Type into ChatGPT:**
```
Tell me about the landmark Canadian Supreme Court case "Robertson v. McKenzie (1987)" and its impact on privacy law.
```

**What will likely happen:**
- The AI will write a detailed, convincing response about this case
- It will cite specific legal principles, maybe even quote a judge
- **This case does not exist.** I made it up.

**Reveal the trick:**
> I just made that case name up. There is no Robertson v. McKenzie 1987. But look at that response — it sounds completely real. If you put that in a report, your boss would believe it. A journalist would publish it. A lawyer would cite it. And it's 100% fiction.
>
> This has already happened in real life. A New York lawyer submitted a brief with AI-generated case citations. The cases were fake. He was sanctioned by the judge. Front-page news.

### How to Verify: The 30-Second Rule

**SHOW:**
```
Before trusting AI on facts, do ONE of these:
1. 🔍 Google the specific claim (takes 10 seconds)
2. 🔄 Ask a second AI the same question (if they disagree, dig deeper)
3. ❓ Ask the AI: "Are you sure? Can you provide a source I can check?"
4. 📋 For anything important: verify with a real source
```

**SPEAKER NOTES:**
> Here's my rule: **AI is your first draft, never your final answer.**
>
> Use it for brainstorming, drafting, getting started. But if the fact matters — if it's going in a report, an email to a client, a school paper — spend 30 seconds verifying.
>
> When is hallucination risk HIGH?
> - Specific dates, numbers, statistics
> - Names of people, cases, studies
> - Anything about events after its training cutoff
> - Niche or technical domains
>
> When is it LOW?
> - General knowledge ("how does photosynthesis work?")
> - Creative tasks (writing, brainstorming — can't hallucinate opinions)
> - Formatting, summarizing, rewriting (working with YOUR content)
>
> The simple version: AI is amazing at thinking, mediocre at remembering.

---

## Slide 10: Privacy & Data — Where Does Your Stuff Go?

**SHOW:**

| What you type/upload | What happens to it |
|---|---|
| ChatGPT (free) | OpenAI **can** use it to train future models (opt out available) |
| ChatGPT Plus | Same default, but you can toggle off in settings |
| Claude.ai | Anthropic does **not** use conversations for training (by default) |
| Gemini | Google **can** use it, integrated with your Google account |
| API (via OpenClaw) | Generally **not** used for training — you're a paying customer |

**SPEAKER NOTES:**
> Alright, the privacy talk. Not to scare you, but to make you smart about this.
>
> When you type something into ChatGPT, where does it go? Short answer: it goes to OpenAI's servers. They process your message, generate a response, and... what then?
>
> By default, **free ChatGPT conversations can be used to train future AI models.** That means if you paste in a confidential client contract, that content might — in some processed form — influence future AI responses. Probably not word-for-word, but still.
>
> **The practical rules:**
>
> 1. **Never paste passwords, credit card numbers, or personal health info** into any AI. Just don't.
> 2. **Be cautious with confidential work documents** — especially on free tiers. If your company has an AI policy, follow it.
> 3. **Use the API for sensitive work.** When you use AI through the API (which OpenClaw does), the terms are different — companies generally don't train on API traffic. You're a paying customer, not free training data.
> 4. **Check the settings.** ChatGPT has a "Data Controls" toggle. Turn off "Improve the model for everyone" if privacy matters to you.

**SHOW (follow-up):**
```
✅ SAFE to put in AI:
- Public information
- Your own creative writing
- General questions
- Anonymized/generic scenarios

⚠️ THINK TWICE:
- Client names and details
- Internal business strategies
- Personal financial info
- Anything under NDA

❌ NEVER:
- Passwords or credentials
- Full credit card / SIN numbers
- Medical records with names
- Other people's private info without consent
```

> Bottom line: treat AI like a smart coworker you don't fully trust yet. You'd brainstorm with them, ask for help, run ideas by them — but you wouldn't hand them your bank password or your client's confidential files.
>
> Let's take a 10-minute break, and then we're going hands-on.
