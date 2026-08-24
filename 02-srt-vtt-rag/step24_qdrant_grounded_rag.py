import os
import sys
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print()
    print("=" * 70)
    print("ERROR — OPENAI_API_KEY NOT FOUND")
    print("=" * 70)
    print()
    print("Create a .env file in this project folder:")
    print()
    print("OPENAI_API_KEY=your_actual_api_key")
    print()
    print("Then run the script again.")
    sys.exit(1)


client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# CONFIGURATION
# ============================================================

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "srt_transcript"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

TOP_K = 3
SIMILARITY_THRESHOLD = 0.40


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("STEP 24 — QDRANT GROUNDED RAG")
print("=" * 70)


# ============================================================
# CHECK QDRANT
# ============================================================

print()
print("Checking Qdrant...")

try:

    response = requests.get(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
        timeout=5
    )

except requests.RequestException as e:

    print()
    print("ERROR — Cannot connect to Qdrant.")
    print(e)
    sys.exit(1)


if response.status_code != 200:

    print()
    print(f"ERROR — Collection '{COLLECTION_NAME}' not found.")
    print()
    print("Run Step 22 first:")
    print()
    print("python step22_qdrant_store.py")
    sys.exit(1)


collection_info = response.json()["result"]


# ============================================================
# QDRANT STATUS
# ============================================================

print()
print("=" * 70)
print("QDRANT CONNECTION")
print("=" * 70)

print(f"Qdrant URL : {QDRANT_URL}")
print(f"Collection : {COLLECTION_NAME}")
print(f"Status     : {collection_info.get('status')}")
print(f"Points     : {collection_info.get('points_count')}")


# ============================================================
# ASK QUESTION
# ============================================================

question = input(
    "\nAsk a question about the transcript: "
).strip()


if not question:

    print("No question provided.")
    sys.exit(0)


print()
print("=" * 70)
print("USER QUESTION")
print("=" * 70)

print(question)


# ============================================================
# GENERATE QUERY EMBEDDING
# ============================================================

print()
print("Generating query embedding...")

try:

    embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )

except Exception as e:

    print()
    print("ERROR — Failed to generate embedding.")
    print(e)
    sys.exit(1)


query_vector = embedding_response.data[0].embedding

print(f"Query vector size: {len(query_vector)}")


# ============================================================
# SEARCH QDRANT
# ============================================================

print()
print("Searching Qdrant...")


search_response = requests.post(
    f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
    json={
        "vector": query_vector,
        "limit": TOP_K,
        "with_payload": True
    },
    timeout=30
)


if search_response.status_code != 200:

    print()
    print("ERROR — Qdrant search failed.")
    print(search_response.text)
    sys.exit(1)


results = search_response.json()["result"]

print()
print("=" * 70)
print("RAW QDRANT SEARCH PAYLOAD")
print("=" * 70)

print(
    json.dumps(
        results[0],
        indent=2
    )
)


# ============================================================
# NO RESULTS
# ============================================================

if not results:

    print()
    print("=" * 70)
    print("NO RESULTS")
    print("=" * 70)

    print("Qdrant returned no matching chunks.")
    sys.exit(0)


# ============================================================
# DISPLAY RETRIEVED CHUNKS
# ============================================================

print()
print("=" * 70)
print("QDRANT RETRIEVAL RESULTS")
print("=" * 70)


for rank, result in enumerate(results, start=1):

    payload = result.get("payload", {})

    metadata = payload.get("metadata", {})

    # IMPORTANT:
    # Step 22 stores transcript under "text"
    text = payload.get("text", "")

    print()
    print("-" * 70)

    print(f"Rank       : {rank}")
    print(f"Point ID   : {result.get('id')}")
    print(f"Chunk ID   : {metadata.get('chunk_id')}")
    print(f"Similarity : {result.get('score', 0):.4f}")

    print()
    print("Timestamp:")
    print(
        f"{metadata.get('start', '')} -> "
        f"{metadata.get('end', '')}"
    )

    print()
    print("Text:")
    print(text[:1000])

    if text:
        print(text[:1000])
    else:
        print("[ERROR] Transcript text is empty.")


# ============================================================
# RELEVANCE CHECK
# ============================================================

best_score = results[0]["score"]


print()
print("=" * 70)
print("RETRIEVAL DECISION")
print("=" * 70)

print(f"Best Similarity Score : {best_score:.4f}")
print(f"Threshold             : {SIMILARITY_THRESHOLD:.2f}")


if best_score < SIMILARITY_THRESHOLD:

    print("Decision              : REJECT")

    print()
    print(
        "This question is outside the scope "
        "of the provided transcript."
    )

    sys.exit(0)


print("Decision              : ACCEPT")


# ============================================================
# BUILD CONTEXT
# ============================================================

context_parts = []


for rank, result in enumerate(results, start=1):

    payload = result.get("payload", {})

    metadata = payload.get("metadata", {})


    # IMPORTANT:
    # Qdrant payload uses "text"    
    text = payload.get("text", "")

    chunk_id = metadata.get(
        "chunk_id",
        "unknown"
    )

    start = metadata.get(
        "start",
        ""
    )

    end = metadata.get(
        "end",
        ""
    )
    if not text:
        print(
            f"WARNING: Empty transcript text for {chunk_id}"
        )


    context_parts.append(
        f"""
SOURCE {rank}

Chunk ID:
{chunk_id}

Timestamp:
{start} -> {end}

Similarity:
{result['score']:.4f}

Transcript:
{text}
""".strip()
    )


context = "\n\n".join(context_parts)


# ============================================================
# DISPLAY CONTEXT
# ============================================================

print()
print("=" * 70)
print("CONTEXT SENT TO LLM")
print("=" * 70)

print(context)


# ============================================================
# GROUNDED SYSTEM PROMPT
# ============================================================

system_prompt = """
You are a grounded RAG assistant.

Your job is to answer the user's question using ONLY
the provided transcript context.

STRICT RULES:

1. Use only information contained in the transcript context.

2. Do not use your own general knowledge.

3. Do not invent or assume facts.

4. If the transcript does not contain enough information,
   respond exactly:

   The transcript does not provide enough information to answer this.

5. Give a concise and direct answer.

6. If the answer is supported by multiple sources,
   combine them carefully.

7. Do not mention information that is not present
   in the transcript.

8. Do not answer based only on similarity scores.
   The actual transcript text must support the answer.
"""


# ============================================================
# USER PROMPT
# ============================================================

user_prompt = f"""
USER QUESTION:
{question}

TRANSCRIPT CONTEXT:
{context}
"""


# ============================================================
# GENERATE GROUNDED ANSWER
# ============================================================

print()
print("Generating grounded answer...")


try:

    chat_response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

except Exception as e:

    print()
    print("ERROR — LLM request failed.")
    print(e)
    sys.exit(1)


answer = chat_response.choices[0].message.content


# ============================================================
# FINAL ANSWER
# ============================================================

print()
print("=" * 70)
print("FINAL GROUNDED ANSWER")
print("=" * 70)

print()
print(answer)


# ============================================================
# SOURCES
# ============================================================

print()
print("=" * 70)
print("SOURCES")
print("=" * 70)


for rank, result in enumerate(results, start=1):

    payload = result.get("payload", {})

    metadata = payload.get("metadata", {})

    print(
        f"[{rank}] "
        f"{metadata.get('chunk_id')} | "
        f"{metadata.get('start')} -> "
        f"{metadata.get('end')} | "
        f"Similarity: "
        f"{result['score']:.4f}"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("STEP 24 COMPLETE")
print("=" * 70)