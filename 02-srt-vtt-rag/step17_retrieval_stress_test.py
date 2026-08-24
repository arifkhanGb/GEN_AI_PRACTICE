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

INPUT_PATH = Path(
    "output/embedded_chunks.json"
)

EMBEDDING_MODEL = "text-embedding-3-small"

RELEVANCE_THRESHOLD = 0.40


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents(json_path):
    """
    Load embedded RAG chunks from JSON.
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
    Convert a question into an embedding vector.
    """

    url = "https://api.openai.com/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": EMBEDDING_MODEL,
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
    Calculate cosine similarity.
    """

    dot_product = sum(
        a * b
        for a, b in zip(
            vector_a,
            vector_b
        )
    )

    magnitude_a = sum(
        a * a
        for a in vector_a
    ) ** 0.5

    magnitude_b = sum(
        b * b
        for b in vector_b
    ) ** 0.5

    if (
        magnitude_a == 0
        or magnitude_b == 0
    ):
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


# ============================================================
# FIND BEST MATCH
# ============================================================

def find_best_match(
    question,
    documents
):
    """
    Find the highest similarity score
    for a question.
    """

    query_embedding = create_embedding(
        question
    )

    best_score = -1.0

    best_chunk = None

    for document in documents:

        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        if score > best_score:

            best_score = score

            best_chunk = document[
                "chunk_id"
            ]

    return best_score, best_chunk


# ============================================================
# CLASSIFY RESULT
# ============================================================

def classify(
    score,
    expected
):
    """
    Classify retrieval result using
    the selected threshold.
    """

    predicted_relevant = (
        score >= RELEVANCE_THRESHOLD
    )

    actual_relevant = (
        expected == "relevant"
    )

    if (
        predicted_relevant
        and actual_relevant
    ):
        return "TP"

    if (
        not predicted_relevant
        and not actual_relevant
    ):
        return "TN"

    if (
        predicted_relevant
        and not actual_relevant
    ):
        return "FP"

    return "FN"


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

QUESTIONS = [

    # ========================================================
    # DIRECT RELEVANT
    # ========================================================

    {
        "question": "What are native components?",
        "expected": "relevant"
    },

    {
        "question": "What is a native component?",
        "expected": "relevant"
    },

    {
        "question": "Which operating systems provide native components?",
        "expected": "relevant"
    },

    {
        "question": "Where do native components exist?",
        "expected": "relevant"
    },

    {
        "question": "Which languages are used to write native components?",
        "expected": "relevant"
    },

    {
        "question": "What languages are used for native components?",
        "expected": "relevant"
    },

    {
        "question": "Are native components directly used in React Native JavaScript code?",
        "expected": "relevant"
    },

    {
        "question": "Can React Native developers directly write native components?",
        "expected": "relevant"
    },

    # ========================================================
    # CORE COMPONENT QUESTIONS
    # ========================================================

    {
        "question": "What are core components?",
        "expected": "relevant"
    },

    {
        "question": "What are some examples of core components?",
        "expected": "relevant"
    },

    {
        "question": "Which components are provided by React Native?",
        "expected": "relevant"
    },

    {
        "question": "Give examples of React Native core components.",
        "expected": "relevant"
    },

    # ========================================================
    # COMPARISON QUESTIONS
    # ========================================================

    {
        "question": "What is the difference between native and core components?",
        "expected": "relevant"
    },

    {
        "question": "How are native components different from core components?",
        "expected": "relevant"
    },

    # ========================================================
    # POSSIBLY AMBIGUOUS
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
    },

    # ========================================================
    # IRRELEVANT
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

    {
        "question": "How does Spring Boot work?",
        "expected": "irrelevant"
    },

    {
        "question": "What is Java?",
        "expected": "irrelevant"
    },

    {
        "question": "What is MongoDB?",
        "expected": "irrelevant"
    },

    {
        "question": "How do I deploy an application to AWS?",
        "expected": "irrelevant"
    },

    {
        "question": "How does Kubernetes work?",
        "expected": "irrelevant"
    },

    {
        "question": "How do I create a PostgreSQL database?",
        "expected": "irrelevant"
    }
]


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    documents = load_documents(
        INPUT_PATH
    )

    print(
        f"Loaded {len(documents)} documents."
    )

    print()
    print("=" * 90)
    print("STEP 17 — RETRIEVAL STRESS TEST")
    print("=" * 90)

    print(
        f"Relevance Threshold : "
        f"{RELEVANCE_THRESHOLD:.2f}"
    )

    print(
        f"Evaluation Questions: "
        f"{len(QUESTIONS)}"
    )

    results = []

    # ========================================================
    # RUN QUESTIONS
    # ========================================================

    for index, item in enumerate(
        QUESTIONS,
        start=1
    ):

        question = item["question"]

        expected = item["expected"]

        print()
        print("-" * 90)

        print(
            f"[{index}/{len(QUESTIONS)}]"
        )

        print(
            f"Question : {question}"
        )

        print(
            f"Expected : {expected}"
        )

        score, chunk_id = find_best_match(
            question,
            documents
        )

        classification = classify(
            score,
            expected
        )

        accepted = (
            score >= RELEVANCE_THRESHOLD
        )

        print(
            f"Best Chunk: {chunk_id}"
        )

        print(
            f"Score     : {score:.4f}"
        )

        print(
            f"Decision  : "
            f"{'ACCEPT' if accepted else 'REJECT'}"
        )

        print(
            f"Result    : {classification}"
        )

        results.append({
            "question": question,
            "expected": expected,
            "score": score,
            "chunk_id": chunk_id,
            "classification": classification
        })

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    tp = sum(
        1
        for result in results
        if result["classification"] == "TP"
    )

    tn = sum(
        1
        for result in results
        if result["classification"] == "TN"
    )

    fp = sum(
        1
        for result in results
        if result["classification"] == "FP"
    )

    fn = sum(
        1
        for result in results
        if result["classification"] == "FN"
    )

    # ========================================================
    # METRICS
    # ========================================================

    if tp + fp > 0:

        precision = tp / (
            tp + fp
        )

    else:

        precision = 0.0

    if tp + fn > 0:

        recall = tp / (
            tp + fn
        )

    else:

        recall = 0.0

    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (
                precision + recall
            )
        )

    else:

        f1 = 0.0

    total = len(results)

    accuracy = (
        tp + tn
    ) / total

    negative_questions = sum(
        1
        for result in results
        if result["expected"] == "irrelevant"
    )

    correctly_rejected = tn

    if negative_questions > 0:

        rejection_rate = (
            correctly_rejected
            / negative_questions
        )

    else:

        rejection_rate = 0.0

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 90)
    print("STEP 17 — STRESS TEST SUMMARY")
    print("=" * 90)

    print()

    print(
        f"Threshold        : "
        f"{RELEVANCE_THRESHOLD:.2f}"
    )

    print(
        f"Total Questions  : "
        f"{total}"
    )

    print(
        f"True Positive    : "
        f"{tp}"
    )

    print(
        f"True Negative    : "
        f"{tn}"
    )

    print(
        f"False Positive   : "
        f"{fp}"
    )

    print(
        f"False Negative   : "
        f"{fn}"
    )

    print()

    print(
        f"Precision        : "
        f"{precision:.4f}"
    )

    print(
        f"Recall           : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score         : "
        f"{f1:.4f}"
    )

    print(
        f"Accuracy         : "
        f"{accuracy:.4f}"
    )

    print(
        f"Rejection Rate   : "
        f"{rejection_rate:.4f}"
    )

    # ========================================================
    # AMBIGUOUS QUESTIONS
    # ========================================================

    ambiguous_results = [
        result
        for result in results
        if result["expected"] == "ambiguous"
    ]

    print()
    print("=" * 90)
    print("AMBIGUOUS QUESTIONS")
    print("=" * 90)

    if not ambiguous_results:

        print("No ambiguous questions.")

    else:

        for result in ambiguous_results:

            print(
                f"Score={result['score']:.4f} | "
                f"{result['question']}"
            )

    # ========================================================
    # FALSE POSITIVES
    # ========================================================

    false_positives = [
        result
        for result in results
        if result["classification"] == "FP"
    ]

    print()
    print("=" * 90)
    print("FALSE POSITIVES")
    print("=" * 90)

    if not false_positives:

        print("None.")

    else:

        for result in false_positives:

            print(
                f"Score={result['score']:.4f} | "
                f"{result['question']}"
            )

    # ========================================================
    # FALSE NEGATIVES
    # ========================================================

    false_negatives = [
        result
        for result in results
        if result["classification"] == "FN"
    ]

    print()
    print("=" * 90)
    print("FALSE NEGATIVES")
    print("=" * 90)

    if not false_negatives:

        print("None.")

    else:

        for result in false_negatives:

            print(
                f"Score={result['score']:.4f} | "
                f"{result['question']}"
            )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    print()
    print("=" * 90)
    print("THRESHOLD DECISION")
    print("=" * 90)

    if (
        fp == 0
        and fn == 0
    ):

        print()
        print(
            f"PASS — Threshold "
            f"{RELEVANCE_THRESHOLD:.2f} "
            f"correctly classified all "
            f"clearly relevant and irrelevant "
            f"questions in this test set."
        )

    elif fp > 0:

        print()
        print(
            "WARNING — Some irrelevant questions "
            "are being accepted."
        )

    elif fn > 0:

        print()
        print(
            "WARNING — Some relevant questions "
            "are being rejected."
        )

    print()
    print(
        "Note: Ambiguous questions are displayed "
        "separately and are not included in the "
        "TP/TN/FP/FN classification."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()