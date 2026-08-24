import os
import sys
import requests
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print()
    print("=" * 70)
    print("ERROR — OPENAI_API_KEY NOT FOUND")
    print("=" * 70)
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
# TEST QUESTIONS
# ============================================================

QUESTIONS = [
    "What are native components?",
    "What operating systems provide native components?",
    "What is the difference between native components and core components?",
    "What languages are mentioned for iOS development?",
    "What examples of native components are mentioned?",
    "What are core components?",
    "What is React Native?",
    "What is the capital of France?"
]


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("STEP 25 — RAG EVALUATION")
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
    print("Run:")
    print("python step22_qdrant_store.py")
    sys.exit(1)


collection_info = response.json()["result"]

print()
print("=" * 70)
print("QDRANT CONNECTION")
print("=" * 70)

print(f"URL        : {QDRANT_URL}")
print(f"Collection : {COLLECTION_NAME}")
print(f"Status     : {collection_info.get('status')}")
print(f"Points     : {collection_info.get('points_count')}")


# ============================================================
# EVALUATION COUNTERS
# ============================================================

total_questions = len(QUESTIONS)
accepted_questions = 0
rejected_questions = 0
empty_context_questions = 0


# ============================================================
# PROCESS QUESTIONS
# ============================================================

for question_number, question in enumerate(
    QUESTIONS,
    start=1
):

    print()
    print()
    print("#" * 70)
    print(
        f"QUESTION {question_number}/{total_questions}"
    )
    print("#" * 70)

    print()
    print("Question:")
    print(question)


    # ========================================================
    # GENERATE QUERY EMBEDDING
    # ========================================================

    print()
    print("Generating query embedding...")

    try:

        embedding_response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=question
        )

    except Exception as e:

        print()
        print("ERROR — Embedding generation failed.")
        print(e)
        continue

    query_vector = (
        embedding_response
        .data[0]
        .embedding
    )

    print(
        f"Query vector size: {len(query_vector)}"
    )


    # ========================================================
    # SEARCH QDRANT
    # ========================================================

    print()
    print("Searching Qdrant...")

    search_response = requests.post(
        f"{QDRANT_URL}/collections/"
        f"{COLLECTION_NAME}/points/search",

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
        continue


    results = search_response.json()["result"]


    # ========================================================
    # NO RESULTS
    # ========================================================

    if not results:

        print()
        print("NO RESULTS FROM QDRANT.")
        rejected_questions += 1
        continue


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print()
    print("-" * 70)
    print("RETRIEVED CHUNKS")
    print("-" * 70)


    for rank, result in enumerate(
        results,
        start=1
    ):

        payload = result.get(
            "payload",
            {}
        )

        text = payload.get(
            "text",
            ""
        )

        metadata = payload.get(
            "metadata",
            {}
        )

        chunk_id = payload.get(
            "chunk_id"
        )

        score = result.get(
            "score",
            0
        )

        print()
        print(
            f"Rank       : {rank}"
        )

        print(
            f"Chunk ID   : {chunk_id}"
        )

        print(
            f"Similarity : {score:.4f}"
        )

        print(
            f"Timestamp  : "
            f"{metadata.get('start', '')}"
            f" -> "
            f"{metadata.get('end', '')}"
        )

        print(
            f"Text chars : {len(text)}"
        )


    # ========================================================
    # RELEVANCE CHECK
    # ========================================================

    best_score = results[0]["score"]


    print()
    print("-" * 70)
    print("RELEVANCE DECISION")
    print("-" * 70)

    print(
        f"Best score : {best_score:.4f}"
    )

    print(
        f"Threshold  : "
        f"{SIMILARITY_THRESHOLD:.2f}"
    )


    if best_score < SIMILARITY_THRESHOLD:

        print(
            "Decision   : REJECT"
        )

        rejected_questions += 1

        continue


    print(
        "Decision   : ACCEPT"
    )

    accepted_questions += 1


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context_parts = []


    for rank, result in enumerate(
        results,
        start=1
    ):

        payload = result.get(
            "payload",
            {}
        )

        text = payload.get(
            "text",
            ""
        )

        metadata = payload.get(
            "metadata",
            {}
        )

        chunk_id = payload.get(
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


    context = "\n\n".join(
        context_parts
    )


    # ========================================================
    # CHECK EMPTY CONTEXT
    # ========================================================

    if not context.strip():

        print()
        print(
            "WARNING — Context is empty."
        )

        empty_context_questions += 1

        continue


    # ========================================================
    # GENERATE GROUNDED ANSWER
    # ========================================================

    system_prompt = """
You are a grounded RAG assistant.

Answer the user's question ONLY using
the provided transcript context.

STRICT RULES:

1. Use only information contained in the transcript.

2. Do not use outside knowledge.

3. Do not invent facts.

4. If the transcript does not provide enough
   information, respond exactly:

The transcript does not provide enough information to answer this.

5. Keep the answer concise.

6. Do not answer based only on similarity scores.

7. The actual transcript text must support
   every factual statement.
"""


    user_prompt = f"""
USER QUESTION:

{question}


TRANSCRIPT CONTEXT:

{context}
"""


    print()
    print("Generating grounded answer...")


    try:

        chat_response = (
            client
            .chat
            .completions
            .create(

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
        )

    except Exception as e:

        print()
        print(
            "ERROR — LLM request failed."
        )

        print(e)

        continue


    answer = (
        chat_response
        .choices[0]
        .message
        .content
    )


    # ========================================================
    # ANSWER
    # ========================================================

    print()
    print("-" * 70)
    print("GROUNDED ANSWER")
    print("-" * 70)

    print()
    print(answer)


# ============================================================
# FINAL EVALUATION
# ============================================================

print()
print()
print("=" * 70)
print("STEP 25 — EVALUATION SUMMARY")
print("=" * 70)

print()
print(
    f"Total questions       : "
    f"{total_questions}"
)

print(
    f"Accepted by retrieval : "
    f"{accepted_questions}"
)

print(
    f"Rejected by retrieval : "
    f"{rejected_questions}"
)

print(
    f"Empty contexts        : "
    f"{empty_context_questions}"
)


# ============================================================
# RETRIEVAL ACCEPTANCE RATE
# ============================================================

if total_questions > 0:

    acceptance_rate = (
        accepted_questions
        / total_questions
    ) * 100

else:

    acceptance_rate = 0


print(
    f"Acceptance rate       : "
    f"{acceptance_rate:.2f}%"
)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)
print("STEP 25 COMPLETE")
print("=" * 70)

if empty_context_questions == 0:

    print()
    print(
        "SUCCESS — No empty retrieval contexts detected."
    )

else:

    print()
    print(
        "WARNING — Empty retrieval contexts detected."
    )