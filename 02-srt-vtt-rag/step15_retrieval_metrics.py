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
    # RELEVANT QUESTIONS
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
            "chunk_0003"
        ]
    },

    {
        "question": "What are some examples of core components?",
        "expected": "relevant",
        "relevant_chunks": [
            "chunk_0003"
        ]
    },


    # ========================================================
    # IRRELEVANT QUESTIONS
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
# HIT@K
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
# RECIPROCAL RANK / MRR
# ============================================================

def calculate_reciprocal_rank(
    retrieved_chunks,
    relevant_chunks
):
    """
    Reciprocal Rank:

    1 / rank of the first relevant chunk.

    Rank 1 -> 1.0
    Rank 2 -> 0.5
    Rank 3 -> 0.3333
    No relevant result -> 0
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
# PRECISION@K
# ============================================================

def calculate_precision_at_k(
    retrieved_chunks,
    relevant_chunks,
    k=3
):
    """
    Precision@K:

    Number of relevant chunks in top-K
    divided by K.
    """

    top_results = retrieved_chunks[:k]

    if not top_results:
        return 0.0

    relevant_ids = set(
        relevant_chunks
    )

    hits = sum(
        1
        for result in top_results
        if result["chunk_id"] in relevant_ids
    )

    return hits / k


# ============================================================
# SCORE THRESHOLD
# ============================================================

def is_relevant_score(
    score,
    threshold=0.50
):
    """
    Determine whether a similarity score
    is considered relevant.
    """

    return score >= threshold


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(
    item,
    documents,
    top_k=3
):
    """
    Evaluate one question and return
    retrieval metrics.
    """

    question = item["question"]
    expected = item["expected"]
    relevant_chunks = item["relevant_chunks"]

    print()
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    retrieved = retrieve_top_k(
        question,
        documents,
        top_k=top_k
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print(f"TOP {top_k} RESULTS")
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

    # --------------------------------------------------------
    # Calculate Hit@K
    # --------------------------------------------------------

    hit_at_k = calculate_hit_at_k(
        retrieved,
        relevant_chunks
    )

    # --------------------------------------------------------
    # Calculate Reciprocal Rank
    # --------------------------------------------------------

    reciprocal_rank = calculate_reciprocal_rank(
        retrieved,
        relevant_chunks
    )

    # --------------------------------------------------------
    # Calculate Precision@K
    # --------------------------------------------------------

    precision_at_k = calculate_precision_at_k(
        retrieved,
        relevant_chunks,
        k=top_k
    )

    # --------------------------------------------------------
    # Maximum score
    # --------------------------------------------------------

    max_score = (
        retrieved[0]["score"]
        if retrieved
        else 0.0
    )

    print()
    print(
        f"Hit@{top_k:<10}: {hit_at_k}"
    )

    print(
        f"Reciprocal Rank : {reciprocal_rank:.4f}"
    )

    print(
        f"Precision@{top_k:<6}: "
        f"{precision_at_k:.4f}"
    )

    print(
        f"Max Score       : {max_score:.4f}"
    )

    return {
        "question": question,
        "expected": expected,
        "hit_at_k": hit_at_k,
        "reciprocal_rank": reciprocal_rank,
        "precision_at_k": precision_at_k,
        "max_score": max_score
    }


# ============================================================
# NEGATIVE QUESTION EVALUATION
# ============================================================

def evaluate_negative_questions(
    evaluation_results,
    threshold=0.50
):
    """
    Check whether irrelevant questions
    are correctly rejected.

    A negative question is correctly rejected
    when its maximum similarity score is
    below the relevance threshold.
    """

    negative_results = []

    for result in evaluation_results:

        if result["expected"] != "irrelevant":
            continue

        rejected = not is_relevant_score(
            result["max_score"],
            threshold
        )

        negative_results.append({
            "question": result["question"],
            "max_score": result["max_score"],
            "rejected": rejected
        })

    return negative_results


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load embedded documents
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
    # Run evaluation
    # --------------------------------------------------------

    evaluation_results = []

    for item in QUESTIONS:

        result = evaluate_question(
            item,
            documents,
            top_k=3
        )

        evaluation_results.append(
            result
        )

    # --------------------------------------------------------
    # Evaluate negative questions
    # --------------------------------------------------------

    negative_results = evaluate_negative_questions(
        evaluation_results,
        threshold=0.50
    )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    total_questions = len(
        evaluation_results
    )

    total_hits = sum(
        result["hit_at_k"]
        for result in evaluation_results
    )

    total_reciprocal_rank = sum(
        result["reciprocal_rank"]
        for result in evaluation_results
    )

    total_precision = sum(
        result["precision_at_k"]
        for result in evaluation_results
    )

    hit_rate_at_k = (
        total_hits / total_questions
        if total_questions > 0
        else 0.0
    )

    mrr = (
        total_reciprocal_rank
        / total_questions
        if total_questions > 0
        else 0.0
    )

    average_precision_at_k = (
        total_precision / total_questions
        if total_questions > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Negative rejection metrics
    # --------------------------------------------------------

    total_negative = len(
        negative_results
    )

    correctly_rejected = sum(
        result["rejected"]
        for result in negative_results
    )

    negative_rejection_rate = (
        correctly_rejected / total_negative
        if total_negative > 0
        else 0.0
    )

    # ========================================================
    # RETRIEVAL METRICS
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("RETRIEVAL METRICS")
    print("=" * 70)

    print()

    print(
        f"Total Questions        : "
        f"{total_questions}"
    )

    print(
        f"Total Hits@3           : "
        f"{total_hits}"
    )

    print(
        f"Hit Rate@3             : "
        f"{hit_rate_at_k:.4f}"
    )

    print(
        f"MRR                    : "
        f"{mrr:.4f}"
    )

    print(
        f"Average Precision@3    : "
        f"{average_precision_at_k:.4f}"
    )

    # ========================================================
    # NEGATIVE RETRIEVAL
    # ========================================================

    print()
    print("=" * 70)
    print("NEGATIVE RETRIEVAL")
    print("=" * 70)

    print()

    print(
        f"Negative Questions     : "
        f"{total_negative}"
    )

    print(
        f"Correctly Rejected     : "
        f"{correctly_rejected}"
    )

    print(
        f"Rejection Rate         : "
        f"{negative_rejection_rate:.4f}"
    )

    # ========================================================
    # NEGATIVE QUESTION DETAILS
    # ========================================================

    print()
    print("=" * 70)
    print("NEGATIVE QUESTION DETAILS")
    print("=" * 70)

    print()

    for result in negative_results:

        status = (
            "REJECTED"
            if result["rejected"]
            else "NOT REJECTED"
        )

        print(
            f"{status:<15} "
            f"score={result['max_score']:.4f} "
            f"| {result['question']}"
        )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print()

    print(
        "Hit Rate@3 = whether at least one relevant "
        "chunk appeared in the top 3."
    )

    print(
        "MRR = how high the first relevant chunk "
        "appeared in the ranking."
    )

    print(
        "Precision@3 = how many of the top 3 results "
        "were relevant."
    )

    print(
        "Rejection Rate = how often irrelevant questions "
        "were correctly rejected."
    )

    print(
        "Relevance threshold = 0.50"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()