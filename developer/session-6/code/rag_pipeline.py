"""
Session 6 Hands-On: RAG Pipeline
Cut the Crap — AI Engineer Edition

Build a document Q&A system with:
- Document loading and chunking
- Embeddings with OpenAI
- Vector storage with ChromaDB
- Retrieval + generation with source citations

Requirements:
    pip install chromadb openai langchain-text-splitters tiktoken
    export OPENAI_API_KEY=your-key
"""

import chromadb
from openai import OpenAI

# --- Setup ---
client = OpenAI()
chroma = chromadb.Client()  # In-memory for demo; use PersistentClient for real use


# --- Sample Documents (replace with your own) ---
DOCUMENTS = {
    "employee_handbook.md": """
# Employee Handbook

## Time Off Policy
Full-time employees receive 20 days of paid time off (PTO) per year.
PTO accrues at 1.67 days per month. Unused PTO carries over up to 5 days.
New hires are eligible for PTO after 90 days of employment.

## Remote Work Policy
Employees may work remotely up to 3 days per week with manager approval.
Remote workers must be available during core hours (10am-3pm local time).
A stable internet connection and secure workspace are required.

## Benefits
Health insurance is provided through BlueCross with 80% company coverage.
Dental and vision are optional add-ons at employee cost.
401k matching is available up to 4% of salary after 1 year of employment.
""",
    "product_faq.md": """
# Product FAQ

## What is AcmeCorp Pro?
AcmeCorp Pro is our enterprise SaaS platform for project management.
It includes task tracking, time logging, team collaboration, and reporting.

## Pricing
- Starter: $10/user/month (up to 10 users)
- Business: $25/user/month (up to 100 users)
- Enterprise: Custom pricing with dedicated support

## Integrations
We integrate with Slack, GitHub, Jira, Google Workspace, and Microsoft 365.
API access is available on Business and Enterprise plans.

## Support
Starter: Email support (48hr response)
Business: Email + chat (24hr response)
Enterprise: Dedicated account manager + phone support (4hr response)
""",
    "engineering_standards.md": """
# Engineering Standards

## Code Review Policy
All changes require at least one approval before merging.
Security-sensitive changes require two approvals including a senior engineer.
Reviews should be completed within 24 hours of submission.

## Deployment Process
We deploy to staging every commit to main.
Production deploys happen Tuesday and Thursday at 2pm EST.
Hotfixes can be deployed anytime with VP Engineering approval.

## Incident Response
P1 (service down): All hands, 15-minute response time, CEO notified.
P2 (degraded service): On-call team, 1-hour response time.
P3 (minor issue): Next business day resolution.
""",
}


# --- Step 1: Chunk Documents ---
def chunk_documents(documents: dict, chunk_size: int = 300, overlap: int = 50) -> list:
    """Split documents into chunks with metadata."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for filename, content in documents.items():
        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "id": f"{filename}__chunk_{i}",
                    "text": chunk.strip(),
                    "source": filename,
                    "chunk_index": i,
                }
            )

    return all_chunks


# --- Step 2: Embed and Store ---
def embed_text(text: str) -> list[float]:
    """Get embedding vector for text."""
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def build_index(chunks: list) -> chromadb.Collection:
    """Embed all chunks and store in ChromaDB."""
    collection = chroma.get_or_create_collection(
        name="company_docs", metadata={"hnsw:space": "cosine"}
    )

    print(f"📄 Embedding {len(chunks)} chunks...")
    for chunk in chunks:
        embedding = embed_text(chunk["text"])
        collection.add(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"], "chunk_index": chunk["chunk_index"]}],
        )

    print(f"✅ Indexed {len(chunks)} chunks into ChromaDB")
    return collection


# --- Step 3: Retrieve ---
def retrieve(collection: chromadb.Collection, query: str, n_results: int = 3) -> list:
    """Search for relevant chunks."""
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append(
            {
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i],
            }
        )

    return retrieved


# --- Step 4: Generate Answer ---
def generate_answer(query: str, context_chunks: list) -> str:
    """Generate an answer using retrieved context."""
    # Format context with source references
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based on the provided context.\n"
                    "Rules:\n"
                    "- Only answer based on the provided context\n"
                    "- Cite your sources using [Source N] references\n"
                    "- If the context doesn't contain the answer, say 'I don't have that information'\n"
                    "- Be concise and direct"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# --- Main Pipeline ---
def main():
    print("🔧 RAG Pipeline — Cut the Crap Edition")
    print("=" * 50)

    # Step 1: Chunk
    print("\n📋 Step 1: Chunking documents...")
    chunks = chunk_documents(DOCUMENTS)
    print(f"   Created {len(chunks)} chunks from {len(DOCUMENTS)} documents")

    # Step 2: Index
    print("\n📦 Step 2: Building vector index...")
    collection = build_index(chunks)

    # Step 3-4: Interactive Q&A
    print("\n" + "=" * 50)
    print("🤖 Ask questions about the company docs!")
    print("   Type 'quit' to exit, 'debug' to see retrieved chunks")
    print("=" * 50)

    debug_mode = False

    while True:
        query = input("\n❓ Question: ").strip()

        if query.lower() == "quit":
            break
        if query.lower() == "debug":
            debug_mode = not debug_mode
            print(f"   Debug mode: {'ON' if debug_mode else 'OFF'}")
            continue
        if not query:
            continue

        # Retrieve
        retrieved = retrieve(collection, query, n_results=3)

        if debug_mode:
            print("\n   📎 Retrieved chunks:")
            for i, chunk in enumerate(retrieved, 1):
                print(f"   [{i}] {chunk['source']} (distance: {chunk['distance']:.4f})")
                print(f"       {chunk['text'][:100]}...")

        # Generate
        answer = generate_answer(query, retrieved)
        print(f"\n💬 Answer: {answer}")


if __name__ == "__main__":
    main()
