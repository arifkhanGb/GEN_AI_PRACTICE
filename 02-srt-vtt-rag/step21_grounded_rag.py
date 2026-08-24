import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not loaded."
    )


EMBEDDING_MODEL = "text-embedding-3-small"

CHAT_MODEL = "gpt-4o-mini"

SIMILARITY_THRESHOLD = 0.40

TOP_K = 3

EMBEDDING_URL = "https://api.openai.com/v1/embeddings"

CHAT_URL = "https://api.openai.com/v1/chat/completions"


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents(json_path):

    with json_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data["documents"]


# ============================================================
# CREATE EMBEDDING
# ============================================================

def create_embedding(text):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }

    response = requests.post(
        EMBEDDING_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    return result["data"][0]["embedding"]


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sum(
        a * a
        for a in vector_a
    ) ** 0.5

    magnitude_b = sum(
        b * b
        for b in vector_b
    ) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(question, documents):

    print()
    print("Generating query embedding...")

    query_embedding = create_embedding(question)

    results = []

    for document in documents:

        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        results.append({
            "document": document,
            "score": score
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:TOP_K]


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(retrieved_chunks):

    context_parts = []

    for rank, item in enumerate(
        retrieved_chunks,
        start=1
    ):

        document = item["document"]

        chunk_id = document.get(
            "chunk_id",
            "unknown"
        )

        text = document.get(
            "text",
            ""
        )

        timestamp = document.get(
            "timestamp",
            ""
        )

        context_parts.append(
            f"""
SOURCE {rank}
Chunk ID: {chunk_id}
Timestamp: {timestamp}

{text}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GENERATE GROUNDED ANSWER
# ============================================================

def generate_answer(question, context):

    system_prompt = """
You are a grounded question-answering assistant.

Your job is to answer the user's question using ONLY
the transcript context provided by the user.

Rules:

1. Use only information present in the context.
2. Do not use outside knowledge.
3. Do not invent or assume facts.
4. If the context does not contain enough information,
   say exactly:

   "The transcript does not provide enough information
   to answer this."

5. Give a concise and direct answer.
6. Do not mention embeddings, similarity scores,
   retrieval, or internal RAG processing.
7. Do not refer to yourself as an AI.
"""


    user_prompt = f"""
USER QUESTION:

{question}


TRANSCRIPT CONTEXT:

{context}


Answer the user's question using ONLY the transcript
context above.
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": CHAT_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    }

    response = requests.post(
        CHAT_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    return result["choices"][0]["message"]["content"].strip()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    input_path = Path(
        "output/embedded_chunks.json"
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"File not found: {input_path}"
        )

    documents = load_documents(
        input_path
    )

    print(
        f"Loaded {len(documents)} documents."
    )

    # --------------------------------------------------------
    # Ask question
    # --------------------------------------------------------

    question = input(
        "\nAsk a question about the transcript: "
    ).strip()

    if not question:

        print("Question cannot be empty.")

        return

    print()
    print("=" * 70)
    print("USER QUESTION")
    print("=" * 70)

    print(question)

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    retrieved_chunks = retrieve_chunks(
        question,
        documents
    )

    best_score = retrieved_chunks[0]["score"]

    # --------------------------------------------------------
    # Display retrieved chunks
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RETRIEVED CHUNKS")
    print("=" * 70)

    for rank, item in enumerate(
        retrieved_chunks,
        start=1
    ):

        document = item["document"]

        print()
        print("-" * 70)

        print(
            f"Rank       : {rank}"
        )

        print(
            f"Chunk ID    : {document.get('chunk_id')}"
        )

        print(
            f"Similarity : {item['score']:.4f}"
        )

        print(
            f"Timestamp  : {document.get('timestamp', '')}"
        )

    print()
    print(
        f"Best Similarity Score : {best_score:.4f}"
    )

    # --------------------------------------------------------
    # Relevance gate
    # --------------------------------------------------------

    if best_score < SIMILARITY_THRESHOLD:

        print()
        print("=" * 70)
        print("NO RELEVANT CONTEXT")
        print("=" * 70)

        print(
            "\nThis question is outside the scope "
            "of the provided transcript."
        )

        return

    print()
    print(
        f"Decision : ACCEPT "
        f"(threshold = {SIMILARITY_THRESHOLD:.2f})"
    )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(
        retrieved_chunks
    )

    print()
    print(
        "Context assembled from top "
        f"{len(retrieved_chunks)} chunks."
    )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    print()
    print("Generating grounded answer...")

    answer = generate_answer(
        question,
        context
    )

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print()
    print(answer)

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    for rank, item in enumerate(
        retrieved_chunks,
        start=1
    ):

        document = item["document"]

        print()
        print(
            f"[{rank}] "
            f"{document.get('chunk_id')}"
        )

        print(
            f"Timestamp: "
            f"{document.get('timestamp', '')}"
        )

        print(
            f"Similarity: "
            f"{item['score']:.4f}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()