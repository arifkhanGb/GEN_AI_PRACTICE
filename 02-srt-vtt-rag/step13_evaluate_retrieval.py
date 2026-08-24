import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# LOAD API KEY
# ============================================================

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
    Load the previously generated documents
    containing text + embeddings.
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
# EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(question, documents):
    """
    Calculate similarity between the question
    and every document.

    We return the highest similarity score.
    """

    print()
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)

    # --------------------------------------------------------
    # Convert question → embedding
    # --------------------------------------------------------

    query_embedding = create_embedding(question)

    scores = []

    # --------------------------------------------------------
    # Compare query against every chunk
    # --------------------------------------------------------

    for document in documents:

        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        scores.append({
            "chunk_id": document["chunk_id"],
            "score": score,
            "text": document["page_content"],
            "metadata": document["metadata"]
        })

    # --------------------------------------------------------
    # Sort highest score first
    # --------------------------------------------------------

    scores.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Display top 3
    # --------------------------------------------------------

    print()
    print("TOP RESULTS")
    print("-" * 70)

    for rank, item in enumerate(
        scores[:3],
        start=1
    ):
        metadata = item["metadata"]

        print()
        print(f"Rank       : {rank}")
        print(f"Chunk ID    : {item['chunk_id']}")
        print(f"Similarity : {item['score']:.4f}")

        print(
            f"Timestamp  : "
            f"{metadata['start']} --> {metadata['end']}"
        )

        print(
            f"Subtitles  : "
            f"{metadata['subtitle_start']} --> "
            f"{metadata['subtitle_end']}"
        )

        print(
            f"Tokens     : "
            f"{metadata['token_count']}"
        )

        print(
            f"Characters : "
            f"{len(item['text'])}"
        )

        print()
        print("Text:")
        print(item["text"][:300])

        print("-" * 70)

    # --------------------------------------------------------
    # Highest similarity
    # --------------------------------------------------------

    max_score = scores[0]["score"]

    print()
    print(
        f"MAX SCORE: {max_score:.4f}"
    )

    return max_score


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

    # ========================================================
    # EVALUATION QUESTIONS
    # ========================================================

    questions = [

        # ========================================================
        # CLEARLY RELEVANT
        # ========================================================

        {
            "question": "What are native components?",
            "expected": "relevant"
        },

        {
            "question": "Which operating systems provide native components?",
            "expected": "relevant"
        },

        {
            "question": "Which languages are used to write native components?",
            "expected": "relevant"
        },

        {
            "question": "Are native components directly used in React Native JavaScript code?",
            "expected": "relevant"
        },

        {
            "question": "What are core components?",
            "expected": "relevant"
        },

        {
            "question": "What are some examples of core components?",
            "expected": "relevant"
        },

        # ========================================================
        # CLEARLY IRRELEVANT
        # ========================================================

        {
            "question": "How do I deploy a React Native application to AWS?",
            "expected": "irrelevant"
        },

        {
            "question": "How do I configure Kubernetes?",
            "expected": "irrelevant"
        },

        {
            "question": "How do I create a PostgreSQL database?",
            "expected": "irrelevant"
        },

        {
            "question": "What is Docker?",
            "expected": "irrelevant"
        },

        # ========================================================
        # AMBIGUOUS / HARD QUESTIONS
        # ========================================================

        {
            "question": "What is View?",
            "expected": "ambiguous"
        },

        {
            "question": "What is Button?",
            "expected": "ambiguous"
        },

        {
            "question": "What languages does React Native use?",
            "expected": "ambiguous"
        },

        {
            "question": "How does React Native work internally?",
            "expected": "ambiguous"
        }
    ]

    # ========================================================
    # RUN EVALUATION
    # ========================================================

    results = []

    for item in questions:

        max_score = evaluate_question(
            item["question"],
            documents
        )

        results.append({
            "question": item["question"],
            "expected": item["expected"],
            "max_score": max_score
        })

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print()

    print(
        f"{'Expected':<12}"
        f"{'Score':<12}"
        f"Question"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['expected']:<12}"
            f"{result['max_score']:<12.4f}"
            f"{result['question']}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()