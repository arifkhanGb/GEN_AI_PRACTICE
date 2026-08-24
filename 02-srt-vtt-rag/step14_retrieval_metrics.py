import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not loaded."
    )


# ============================================================
# LOAD EMBEDDED DOCUMENTS
# ============================================================

def load_documents(json_path):
    """
    Load documents containing:
    - page_content
    - metadata
    - embedding
    """

    with json_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data["documents"]


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

def create_embedding(text):
    """
    Convert the user's question into
    an embedding vector.
    """

    url = "https://api.openai.com/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "text-embedding-3-small",
        "input": text
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
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between
    two vectors.
    """

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
# RETRIEVE TOP-K
# ============================================================

def retrieve_top_k(
    question,
    documents,
    top_k=3
):
    """
    Retrieve the top-K chunks based
    on cosine similarity.
    """

    query_embedding = create_embedding(
        question
    )

    scores = []

    for document in documents:

        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        scores.append({
            "chunk_id": document["chunk_id"],
            "score": score
        })

    scores.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scores[:top_k]


# ============================================================
# EVALUATION DATASET
# ============================================================

QUESTIONS = [

    # ========================================================
    # RELEVANT
    # ========================================================

    {
        "question": "What are native components?",
        "expected": "relevant",
        "relevant_chunks": [
            "chunk_0001",
            "chunk_0002",
            "chunk_0003"
        ]
    },

    {
        "question": "Which operating systems provide native components?",
        "expected": "relevant",
        "relevant_chunks": [
            "chunk_0001",
            "chunk_0002"
        ]
    },

    {
        "question": "Which languages are used to write native components?",
        "expected": "relevant",
        "relevant_chunks": [
            "chunk_0002",
            "chunk_0003"
        ]
    },

    {
        "question": "Are native components directly used in React Native JavaScript code?",
        "expected": "relevant",
        "relevant_chunks": [
            "chunk_0002",
            "chunk_0003"
        ]
    },

    {
        "question": "What are core components?",
        "expected": "relevant",
        "relevant_chunks": [
            "chunk_0003",
            "chunk_0004"
        ]
    },

    {
        "question": "What are some examples of core components?",
        "expected": "relevant",
        "relevant_chunks": [
            "chunk_0003",
            "chunk_0004"
        ]
    },


    # ========================================================
    # IRRELEVANT
    # ========================================================

    {
        "question": "How do I deploy a React Native application to AWS?",
        "expected": "irrelevant",
        "relevant_chunks": []
    },

    {
        "question": "How do I configure Kubernetes?",
        "expected": "irrelevant",
        "relevant_chunks": []
    },

    {
        "question": "How do I create a PostgreSQL database?",
        "expected": "irrelevant",
        "relevant_chunks": []
    },

    {
        "question": "What is Docker?",
        "expected": "irrelevant",
        "relevant_chunks": []
    }
]


# ============================================================
# CHECK HIT@K
# ============================================================

def calculate_hit_at_k(
    retrieved_chunks,
    relevant_chunks
):
    """
    Hit@K = 1 if at least one relevant chunk
    appears in the retrieved results.
    Otherwise 0.
    """

    retrieved_ids = {
        item["chunk_id"]
        for item in retrieved_chunks
    }

    relevant_ids = set(
        relevant_chunks
    )

    if retrieved_ids.intersection(
        relevant_ids
    ):
        return 1

    return 0


# ============================================================
# CHECK MRR
# ============================================================

def calculate_reciprocal_rank(
    retrieved_chunks,
    relevant_chunks
):
    """
    Reciprocal Rank:

    1 / rank of the first relevant chunk.

    Example:

    Rank 1 relevant → 1.0
    Rank 2 relevant → 0.5
    Rank 3 relevant → 0.333
    No relevant result → 0
    """

    relevant_ids = set(
        relevant_chunks
    )

    for rank, item in enumerate(
        retrieved_chunks,
        start=1
    ):

        if item["chunk_id"] in relevant_ids:

            return 1 / rank

    return 0.0


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
    # Evaluation
    # --------------------------------------------------------

    evaluation_results = []

    for item in QUESTIONS:

        question = item["question"]

        expected = item["expected"]

        relevant_chunks = item[
            "relevant_chunks"
        ]

        print()
        print("=" * 70)
        print("QUESTION")
        print("=" * 70)

        print(question)

        # ----------------------------------------------------
        # Retrieve
        # ----------------------------------------------------

        retrieved = retrieve_top_k(
            question,
            documents,
            top_k=3
        )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print()
        print("TOP 3 RESULTS")
        print("-" * 70)

        for rank, result in enumerate(
            retrieved,
            start=1
        ):

            print(
                f"Rank {rank}: "
                f"{result['chunk_id']} "
                f"score={result['score']:.4f}"
            )

        # ----------------------------------------------------
        # Hit@3
        # ----------------------------------------------------

        hit_at_3 = calculate_hit_at_k(
            retrieved,
            relevant_chunks
        )

        # ----------------------------------------------------
        # Reciprocal Rank
        # ----------------------------------------------------

        reciprocal_rank = calculate_reciprocal_rank(
            retrieved,
            relevant_chunks
        )

        print()
        print(
            f"Hit@3           : {hit_at_3}"
        )

        print(
            f"Reciprocal Rank : {reciprocal_rank:.4f}"
        )

        evaluation_results.append({
            "question": question,
            "expected": expected,
            "hit_at_3": hit_at_3,
            "reciprocal_rank": reciprocal_rank
        })


    # ========================================================
    # FINAL METRICS
    # ========================================================

    total_questions = len(
        evaluation_results
    )

    total_hits = sum(
        result["hit_at_3"]
        for result in evaluation_results
    )

    total_reciprocal_rank = sum(
        result["reciprocal_rank"]
        for result in evaluation_results
    )

    hit_rate_at_3 = (
        total_hits / total_questions
    )

    mrr = (
        total_reciprocal_rank
        / total_questions
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("RETRIEVAL METRICS")
    print("=" * 70)

    print()

    print(
        f"Total Questions : {total_questions}"
    )

    print(
        f"Total Hits@3    : {total_hits}"
    )

    print(
        f"Hit Rate@3      : {hit_rate_at_3:.4f}"
    )

    print(
        f"MRR             : {mrr:.4f}"
    )

    print()
    print("=" * 70)

    print(
        "Interpretation:"
    )

    print(
        "Hit Rate@3 tells us whether at least one "
        "relevant chunk appeared in the top 3."
    )

    print(
        "MRR tells us how high the first relevant "
        "chunk appeared in the ranking."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()