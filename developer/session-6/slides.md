# Session 6: RAG & Data
## Cut the Crap — AI Engineer Edition

---

## Slide 1: What We're Covering Today

- RAG: what it is and why it matters
- The full pipeline: embed → store → retrieve → generate
- Vector databases: Chroma, Pinecone, pgvector, Weaviate
- RAG vs long context vs fine-tuning — when to use what
- Fine-tuning basics
- Context window management
- **Hands-on:** Build a RAG pipeline from scratch

> **SPEAKER NOTES:**
> "Today is the session that makes AI useful for YOUR data. Everything we've done so far uses the LLM's training data. But what if you want it to answer questions about your company docs, your codebase, your private data? That's RAG — Retrieval-Augmented Generation. It's the most deployed AI pattern in enterprise."

---

## Slide 2: The Problem RAG Solves

**LLMs know a lot, but they don't know YOUR stuff.**

- Your internal docs ❌
- Your product specs ❌
- Your customer data ❌
- Anything after their training cutoff ❌

**Two options:**
1. Fine-tune the model on your data (expensive, slow)
2. **Give it your data at query time** (RAG) ✅

> **SPEAKER NOTES:**
> "When someone at your company asks 'can we make ChatGPT know about our internal docs?' — the answer is RAG. You don't retrain the model. You search your docs for relevant chunks, stuff them into the prompt, and let the LLM answer based on that context. Simple idea, tricky to get right."

---

## Slide 3: RAG Pipeline — The Big Picture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Your Docs   │────▶│   Chunking   │────▶│  Embeddings  │
│  PDF, MD, DB │     │  Split text  │     │  Text → Vec  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │  Vector DB   │
                                          │  Store vecs  │
                                          └──────┬───────┘
                                                  │
     ┌──────────────┐     ┌──────────────┐       │
     │   Response   │◀────│     LLM      │◀──────┘
     │  To user     │     │  Generate    │  + user query
     └──────────────┘     └──────────────┘
```

**Two phases:**
1. **Indexing** (offline): Chunk docs → embed → store in vector DB
2. **Querying** (online): Embed query → search vector DB → feed context to LLM

> **SPEAKER NOTES:**
> "RAG has two phases. First, you index your documents — split them into chunks, convert each chunk into a vector (embedding), and store them in a vector database. You do this once, or on a schedule. Second, when a user asks a question, you embed their question, find the most similar document chunks, and pass those chunks to the LLM as context. The LLM then generates an answer based on those specific chunks."

---

## Slide 4: What Are Embeddings?

```python
from openai import OpenAI
client = OpenAI()

# Turn text into a vector (list of numbers)
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="The cat sat on the mat"
)

vector = response.data[0].embedding
print(f"Dimensions: {len(vector)}")  # 1536
print(f"First 5: {vector[:5]}")      # [0.023, -0.041, 0.018, ...]
```

**Key insight:** Similar meanings → similar vectors

```
"The cat sat on the mat"     → [0.023, -0.041, 0.018, ...]
"A feline rested on a rug"  → [0.025, -0.039, 0.020, ...]  ← CLOSE!
"Stock prices rose today"   → [-0.056, 0.032, -0.011, ...] ← FAR!
```

> **SPEAKER NOTES:**
> "An embedding is just a list of numbers that represents the *meaning* of text. The magic is that similar meanings produce similar numbers. So 'cat on a mat' and 'feline on a rug' are close together in vector space, even though the words are different. This is how we do semantic search — not keyword matching, but meaning matching."

---

## Slide 5: Embedding Models Compared

| Model | Provider | Dimensions | Cost (per 1M tokens) |
|-------|----------|-----------|----------------------|
| `text-embedding-3-small` | OpenAI | 1536 | $0.02 |
| `text-embedding-3-large` | OpenAI | 3072 | $0.13 |
| `voyage-3` | Voyage AI | 1024 | $0.06 |
| `embed-v4.0` | Cohere | 1024 | $0.10 |
| `nomic-embed-text` | Local (Ollama) | 768 | Free |
| `all-MiniLM-L6-v2` | Local (sentence-transformers) | 384 | Free |

> **SPEAKER NOTES:**
> "Embedding models are cheap — way cheaper than generation models. OpenAI's small model is $0.02 per million tokens. That means embedding an entire book costs pennies. For most use cases, the small model is fine. If you need the absolute best retrieval quality, go with text-embedding-3-large or Voyage. And yes, you can run embedding models locally for free with Ollama."

---

## Slide 6: Chunking — The Underrated Step

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = open("big_document.md").read()

# Bad: fixed-size chunks that split mid-sentence
bad_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

# Good: overlap + respect sentence boundaries
good_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,      # Overlap catches context at boundaries
    separators=["\n\n", "\n", ". ", " ", ""],  # Try paragraphs first
)

chunks = good_splitter.split_text(text)
print(f"Split into {len(chunks)} chunks")
```

**Chunking rules of thumb:**
- 200-500 tokens per chunk for most use cases
- Always use overlap (10-20% of chunk size)
- Respect document structure (headers, paragraphs)
- Smaller chunks = more precise retrieval, less context
- Bigger chunks = more context, noisier retrieval

> **SPEAKER NOTES:**
> "Chunking is where most RAG pipelines silently fail. If you chunk poorly — splitting mid-sentence, losing context — your retrieval will suck no matter how good your embedding model is. The RecursiveCharacterTextSplitter tries paragraph breaks first, then sentences, then words. The overlap ensures you don't lose context at chunk boundaries. This is worth spending time on for your specific documents."

---

## Slide 7: Vector Databases

| Database | Type | Best For | Pricing |
|----------|------|----------|---------|
| **Chroma** | Embedded / server | Prototyping, small-medium | Free (open source) |
| **Pinecone** | Managed cloud | Production, zero ops | Free tier, then $/usage |
| **pgvector** | Postgres extension | Already using Postgres | Free (extension) |
| **Weaviate** | Cloud / self-hosted | Hybrid search | Free tier available |
| **Qdrant** | Cloud / self-hosted | Performance-critical | Free tier available |
| **FAISS** | In-memory library | Research, benchmarking | Free (Meta) |

> **SPEAKER NOTES:**
> "For today's hands-on we'll use Chroma because it runs locally with zero setup. For production, I'd recommend pgvector if you already have Postgres — why add another database? Pinecone if you want zero ops. The choice matters less than you think — the chunking and embedding strategy matters way more."

---

## Slide 8: The Full RAG Pipeline in Code

```python
import chromadb
from openai import OpenAI

client = OpenAI()
chroma = chromadb.Client()
collection = chroma.create_collection("my_docs")

# 1. INDEX: Embed and store documents
docs = [
    "Our refund policy allows returns within 30 days.",
    "Premium plans include priority support and API access.",
    "Office hours are Monday-Friday, 9am-5pm EST.",
]

for i, doc in enumerate(docs):
    embedding = client.embeddings.create(
        model="text-embedding-3-small", input=doc
    ).data[0].embedding
    
    collection.add(
        ids=[f"doc_{i}"],
        embeddings=[embedding],
        documents=[doc]
    )

# 2. QUERY: Embed question, search, generate
question = "Can I get a refund?"
q_embedding = client.embeddings.create(
    model="text-embedding-3-small", input=question
).data[0].embedding

results = collection.query(query_embeddings=[q_embedding], n_results=2)
context = "\n".join(results["documents"][0])

# 3. GENERATE: LLM answers using retrieved context
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": f"Answer based on this context:\n{context}"},
        {"role": "user", "content": question}
    ]
)
print(response.choices[0].message.content)
```

> **SPEAKER NOTES:**
> "Here's the full pipeline in 30 lines. Index: embed your docs and store them. Query: embed the question, find similar docs. Generate: pass the retrieved docs as context to the LLM. That's it. This is RAG stripped to its essence. Everything else — reranking, hybrid search, metadata filtering — is optimization on top of this."

---

## Slide 9: RAG Failure Modes (What Goes Wrong)

| Problem | Symptom | Fix |
|---------|---------|-----|
| Bad chunking | Relevant info split across chunks | Better splitter, more overlap |
| Wrong k | Missing relevant docs or too much noise | Tune n_results (start with 3-5) |
| Embedding mismatch | Query finds wrong docs | Try different embedding model |
| No relevant docs | LLM hallucinates an answer | Add "say I don't know" to system prompt |
| Stale data | Outdated answers | Re-index on a schedule |
| Too much context | LLM gets confused | Rerank results, use less |

> **SPEAKER NOTES:**
> "RAG looks simple but there are many ways it fails silently. The worst one: the LLM will confidently answer even when the retrieved docs are irrelevant. Always add 'If the context doesn't contain the answer, say you don't know' to your system prompt. And test your retrieval separately from your generation — if the right chunks aren't being retrieved, no LLM can save you."

---

## Slide 10: Advanced RAG Techniques

**1. Hybrid Search** — Combine vector search + keyword search (BM25)
```python
# Weaviate example
results = collection.query.hybrid(query="refund policy", alpha=0.5)
# alpha=0 → pure keyword, alpha=1 → pure vector
```

**2. Reranking** — Score results with a more powerful model
```python
from cohere import Client
co = Client()
reranked = co.rerank(query="refund", documents=results, model="rerank-v3.5")
```

**3. Query Expansion** — Rephrase the query for better retrieval
```python
# Ask LLM to generate multiple search queries
expanded = llm.invoke("Generate 3 search queries for: Can I get my money back?")
# → ["refund policy", "return process", "money back guarantee"]
```

**4. Contextual Retrieval** — Anthropic's approach: prepend context to chunks
```
Before: "Returns are allowed within 30 days."
After:  "From the Refund Policy document, Section 2: Returns are allowed within 30 days."
```

> **SPEAKER NOTES:**
> "Once basic RAG works, these techniques improve it. Hybrid search is the biggest bang for your buck — combining meaning-based and keyword-based search catches things either misses alone. Reranking uses a specialized model to re-score your results. Query expansion generates multiple search queries from one question. And Anthropic published a 'contextual retrieval' paper — prepend document metadata to each chunk before embedding. These are all worth trying if basic RAG isn't accurate enough."

---

## Slide 11: RAG vs Long Context vs Fine-Tuning

| Approach | When to Use | Cost | Freshness |
|----------|-------------|------|-----------|
| **RAG** | Lots of docs, need freshness | Low-Medium | Real-time |
| **Long Context** | Few docs that fit in context | High (tokens!) | Per-request |
| **Fine-tuning** | Need behavior change, not knowledge | High (training) | Static |

**Decision tree:**
```
Do your docs fit in the context window?
├── Yes → Just stuff them in the prompt (long context)
├── Barely → Try long context first, fall back to RAG
└── No → RAG
    
Do you need the model to ACT differently (tone, format)?
├── Yes → Fine-tuning
└── No → RAG or prompt engineering
```

> **SPEAKER NOTES:**
> "Common question: with 200K context windows, do we still need RAG? Yes, for three reasons. First, cost — sending 200K tokens every request is expensive. Second, accuracy — models actually perform worse with very long contexts (needle-in-a-haystack problem). Third, freshness — RAG lets you update your knowledge base without re-doing anything. Fine-tuning is for changing HOW the model behaves, not WHAT it knows. If you want it to adopt your company's writing style, fine-tune. If you want it to know your product catalog, use RAG."

---

## Slide 12: Fine-Tuning in 60 Seconds

```python
from openai import OpenAI
client = OpenAI()

# 1. Prepare training data (JSONL format)
# training_data.jsonl:
# {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

# 2. Upload file
file = client.files.create(file=open("training_data.jsonl", "rb"), purpose="fine-tune")

# 3. Start fine-tuning
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18"
)

# 4. Use your fine-tuned model
response = client.chat.completions.create(
    model="ft:gpt-4o-mini:my-org::abc123",  # Your fine-tuned model ID
    messages=[{"role": "user", "content": "..."}]
)
```

**Fine-tuning costs:**
- GPT-4o-mini: ~$3/1M training tokens
- GPT-4o: ~$25/1M training tokens
- Need 50-100+ examples minimum

> **SPEAKER NOTES:**
> "Fine-tuning is simpler than people think but more limited than people hope. You prepare examples of ideal input/output pairs, upload them, and OpenAI trains a custom version of the model. It's great for: consistent formatting, specific tone of voice, or domain-specific jargon. It's NOT great for: teaching the model new facts (use RAG), or complex reasoning (use better prompts). Most teams that think they need fine-tuning actually need better prompts or RAG."

---

## Slide 13: Context Window Management

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def manage_context(messages: list, max_tokens: int = 100000) -> list:
    """Keep messages within token budget."""
    total = sum(count_tokens(m["content"]) for m in messages)
    
    # Strategy 1: Drop oldest messages (keep system + last N)
    while total > max_tokens and len(messages) > 2:
        removed = messages.pop(1)  # Remove oldest non-system message
        total -= count_tokens(removed["content"])
    
    return messages

def summarize_and_compress(messages: list, client) -> list:
    """Strategy 2: Summarize old messages."""
    if len(messages) < 10:
        return messages
    
    old = messages[1:-4]  # Keep system prompt and last 4
    summary = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Summarize this conversation:\n{old}"
        }]
    ).choices[0].message.content
    
    return [
        messages[0],  # System prompt
        {"role": "system", "content": f"Previous conversation summary: {summary}"},
        *messages[-4:]  # Last 4 messages
    ]
```

> **SPEAKER NOTES:**
> "Context windows are big now — 128K-200K tokens — but they're not infinite, and they cost money. Two practical strategies: sliding window (drop oldest messages) and summarization (compress old messages into a summary). For RAG, you also need to budget tokens for retrieved context. A good rule: reserve 50% for retrieved docs, 25% for conversation history, 25% for the response."

---

## Slide 14: Hands-On — Build a RAG Pipeline

We're building a document Q&A system:
1. Load and chunk documents
2. Embed with OpenAI
3. Store in Chroma
4. Query with semantic search
5. Generate answers with source citations

**Open:** `session-6/code/rag_pipeline.py`

> **SPEAKER NOTES:**
> "Hands-on time. We're building a complete RAG pipeline. We'll load some sample documents, chunk them, embed them into Chroma, and then build a question-answering interface that cites its sources. Open the code file and let's go."

---

## Slide 15: Recap

✅ RAG = search your docs, stuff into prompt, generate answer
✅ Embeddings turn meaning into numbers for similarity search
✅ Chunking strategy matters more than you think
✅ Start with Chroma (local), move to pgvector/Pinecone for production
✅ Use RAG for knowledge, fine-tuning for behavior
✅ Always add "say I don't know" to prevent hallucinations

**Next session:** Observability, Evals & Security — making sure your AI actually works and doesn't get hacked

> **SPEAKER NOTES:**
> "RAG is the bread and butter of AI engineering. If you remember one thing: garbage in, garbage out. Spend time on your chunking strategy and test your retrieval quality before blaming the LLM. Next time we cover the unglamorous but critical stuff — how to monitor your AI, test it, and keep it secure."
