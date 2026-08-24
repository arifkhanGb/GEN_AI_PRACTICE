import os

import requests
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

COLLECTION_NAME = "srt_transcript"

EMBEDDING_MODEL = "text-embedding-3-small"

TOP_K = 3

SIMILARITY_THRESHOLD = 0.40


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not loaded."
    )


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

def create_query_embedding(question):
    """
    Convert the user's question into
    a 1536-dimensional embedding vector.
    """

    url = (
        "https://api.openai.com/v1/embeddings"
    )

    headers = {
        "Authorization": (
            f"Bearer {OPENAI_API_KEY}"
        ),
        "Content-Type": "application/json"
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "input": question
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    return result["data"][0]["embedding"]


# ============================================================
# SEARCH QDRANT
# ============================================================

def search_qdrant(query_vector):
    """
    Search Qdrant using the query embedding.

    Qdrant performs the cosine similarity
    calculation internally.
    """

    url = (
        f"{QDRANT_URL}/collections/"
        f"{COLLECTION_NAME}/points/search"
    )

    payload = {
        "vector": query_vector,
        "limit": TOP_K,
        "with_payload": True,
        "with_vector": False
    }

    response = requests.post(
        url,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    return result["result"]


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(question, results):
    """
    Display Qdrant retrieval results.
    """

    print()
    print("=" * 70)
    print("USER QUESTION")
    print("=" * 70)

    print(question)

    print()
    print("=" * 70)
    print("QDRANT RETRIEVAL RESULTS")
    print("=" * 70)

    if not results:
        print(
            "No results returned from Qdrant."
        )
        return None

    for rank, result in enumerate(
        results,
        start=1
    ):

        score = result["score"]

        payload = result.get(
            "payload",
            {}
        )

        chunk_id = payload.get(
            "chunk_id",
            "unknown"
        )

        text = payload.get(
            "text",
            ""
        )

        metadata = payload.get(
            "metadata",
            {}
        )

        print()
        print("-" * 70)

        print(
            f"Rank       : {rank}"
        )

        print(
            f"Point ID   : {result['id']}"
        )

        print(
            f"Chunk ID   : {chunk_id}"
        )

        print(
            f"Similarity : {score:.4f}"
        )

        print(
            f"Text       : {text[:500]}"
        )

        print(
            f"Metadata   : {metadata}"
        )

    best_score = results[0]["score"]

    print()
    print("-" * 70)

    print(
        f"Best Similarity Score : "
        f"{best_score:.4f}"
    )

    if best_score >= SIMILARITY_THRESHOLD:

        print(
            f"Decision              : ACCEPT "
            f"(threshold = "
            f"{SIMILARITY_THRESHOLD:.2f})"
        )

    else:

        print(
            f"Decision              : REJECT "
            f"(threshold = "
            f"{SIMILARITY_THRESHOLD:.2f})"
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STEP 23 — QDRANT VECTOR RETRIEVAL")
    print("=" * 70)

    print(
        f"Qdrant URL : {QDRANT_URL}"
    )

    print(
        f"Collection : {COLLECTION_NAME}"
    )

    print(
        f"Top K      : {TOP_K}"
    )

    print(
        f"Threshold  : {SIMILARITY_THRESHOLD}"
    )

    # --------------------------------------------------------
    # Ask question
    # --------------------------------------------------------

    question = input(
        "\nAsk a question about the transcript: "
    ).strip()

    if not question:
        print(
            "Question cannot be empty."
        )
        return

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    print()
    print(
        "Generating query embedding..."
    )

    query_vector = create_query_embedding(
        question
    )

    print(
        f"Query vector size: "
        f"{len(query_vector)}"
    )

    # --------------------------------------------------------
    # Search Qdrant
    # --------------------------------------------------------

    print()
    print(
        "Searching Qdrant..."
    )

    results = search_qdrant(
        query_vector
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    display_results(
        question,
        results
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()