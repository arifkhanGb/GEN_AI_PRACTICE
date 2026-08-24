"""The purpose is:

Find the threshold that best separates relevant and irrelevant questions.

The next code should calculate:
"""

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
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "text-embedding-3-small"

INPUT_PATH = Path(
    "output/embedded_chunks.json"
)

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


# ============================================================
# LOAD EMBEDDED DOCUMENTS
# ============================================================

def load_documents(json_path):
    """
    Load documents containing:

    - page_content
    - metadata
    - embedding
    - chunk_id
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
    Calculate cosine similarity between
    two embedding vectors.
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
# CALCULATE MAX SIMILARITY
# ============================================================

def calculate_max_score(
    question,
    documents
):
    """
    Calculate similarity between the question
    and every document.

    Return the highest similarity score.
    """

    query_embedding = create_embedding(
        question
    )

    max_score = -1.0

    best_chunk = None

    for document in documents:

        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        if score > max_score:

            max_score = score

            best_chunk = document[
                "chunk_id"
            ]

    return max_score, best_chunk


# ============================================================
# CLASSIFY QUESTION
# ============================================================

def classify_score(
    score,
    expected,
    threshold
):
    """
    Classify the retrieval result.

    Relevant question:
        score >= threshold → accepted

    Irrelevant question:
        score < threshold → rejected
    """

    predicted_relevant = (
        score >= threshold
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
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    evaluation_results,
    threshold
):
    """
    Calculate:

    TP
    TN
    FP
    FN
    Precision
    Recall
    F1
    Accuracy
    Rejection Rate
    """

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    negative_questions = 0
    correctly_rejected = 0

    for result in evaluation_results:

        classification = classify_score(
            result["max_score"],
            result["expected"],
            threshold
        )

        if classification == "TP":
            tp += 1

        elif classification == "TN":
            tn += 1

        elif classification == "FP":
            fp += 1

        elif classification == "FN":
            fn += 1

        if result["expected"] == "irrelevant":

            negative_questions += 1

            if classification == "TN":
                correctly_rejected += 1

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if (tp + fp) > 0:

        precision = tp / (
            tp + fp
        )

    else:

        precision = 0.0

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if (tp + fn) > 0:

        recall = tp / (
            tp + fn
        )

    else:

        recall = 0.0

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if (
        precision + recall
    ) > 0:

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

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    total = (
        tp
        + tn
        + fp
        + fn
    )

    if total > 0:

        accuracy = (
            tp + tn
        ) / total

    else:

        accuracy = 0.0

    # --------------------------------------------------------
    # Rejection Rate
    # --------------------------------------------------------

    if negative_questions > 0:

        rejection_rate = (
            correctly_rejected
            / negative_questions
        )

    else:

        rejection_rate = 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "rejection_rate": rejection_rate
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    documents = load_documents(
        INPUT_PATH
    )

    print(
        f"Loaded {len(documents)} documents."
    )

    # ========================================================
    # EVALUATION DATASET
    # ========================================================

    questions = [

        # ====================================================
        # RELEVANT QUESTIONS
        # ====================================================

        {
            "question":
                "What are native components?",
            "expected":
                "relevant"
        },

        {
            "question":
                "Which operating systems provide native components?",
            "expected":
                "relevant"
        },

        {
            "question":
                "Which languages are used to write native components?",
            "expected":
                "relevant"
        },

        {
            "question":
                "Are native components directly used in React Native JavaScript code?",
            "expected":
                "relevant"
        },

        {
            "question":
                "What are core components?",
            "expected":
                "relevant"
        },

        {
            "question":
                "What are some examples of core components?",
            "expected":
                "relevant"
        },

        # ====================================================
        # IRRELEVANT QUESTIONS
        # ====================================================

        {
            "question":
                "How do I deploy a React Native application to AWS?",
            "expected":
                "irrelevant"
        },

        {
            "question":
                "How do I configure Kubernetes?",
            "expected":
                "irrelevant"
        },

        {
            "question":
                "How do I create a PostgreSQL database?",
            "expected":
                "irrelevant"
        },

        {
            "question":
                "What is Docker?",
            "expected":
                "irrelevant"
        }
    ]

    # ========================================================
    # CALCULATE SCORES
    # ========================================================

    print()
    print("=" * 70)
    print("CALCULATING QUERY SIMILARITY SCORES")
    print("=" * 70)

    evaluation_results = []

    for item in questions:

        question = item["question"]

        expected = item["expected"]

        print()
        print(
            f"Question: {question}"
        )

        max_score, best_chunk = (
            calculate_max_score(
                question,
                documents
            )
        )

        print(
            f"Expected : {expected}"
        )

        print(
            f"Best Chunk: {best_chunk}"
        )

        print(
            f"Max Score: {max_score:.4f}"
        )

        evaluation_results.append({

            "question": question,

            "expected": expected,

            "max_score": max_score,

            "best_chunk": best_chunk
        })

    # ========================================================
    # THRESHOLD EVALUATION
    # ========================================================

    print()
    print()
    print("=" * 100)
    print("THRESHOLD EVALUATION")
    print("=" * 100)

    print()

    print(
        f"{'Threshold':<12}"
        f"{'TP':<6}"
        f"{'TN':<6}"
        f"{'FP':<6}"
        f"{'FN':<6}"
        f"{'Precision':<12}"
        f"{'Recall':<10}"
        f"{'F1':<10}"
        f"{'Accuracy':<10}"
        f"{'Reject':<10}"
    )

    print("-" * 100)

    threshold_results = []

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            evaluation_results,
            threshold
        )

        threshold_results.append(
            metrics
        )

        print(
            f"{threshold:<12.2f}"
            f"{metrics['tp']:<6}"
            f"{metrics['tn']:<6}"
            f"{metrics['fp']:<6}"
            f"{metrics['fn']:<6}"
            f"{metrics['precision']:<12.4f}"
            f"{metrics['recall']:<10.4f}"
            f"{metrics['f1']:<10.4f}"
            f"{metrics['accuracy']:<10.4f}"
            f"{metrics['rejection_rate']:<10.4f}"
        )

    # ========================================================
    # FIND BEST THRESHOLD
    # ========================================================

    best_threshold = max(
        threshold_results,
        key=lambda item: (
            item["f1"],
            item["accuracy"],
            item["rejection_rate"]
        )
    )

    print()
    print("=" * 70)
    print("BEST THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold       : "
        f"{best_threshold['threshold']:.2f}"
    )

    print(
        f"Precision       : "
        f"{best_threshold['precision']:.4f}"
    )

    print(
        f"Recall          : "
        f"{best_threshold['recall']:.4f}"
    )

    print(
        f"F1 Score        : "
        f"{best_threshold['f1']:.4f}"
    )

    print(
        f"Accuracy        : "
        f"{best_threshold['accuracy']:.4f}"
    )

    print(
        f"Rejection Rate  : "
        f"{best_threshold['rejection_rate']:.4f}"
    )

    print(
        f"TP              : "
        f"{best_threshold['tp']}"
    )

    print(
        f"TN              : "
        f"{best_threshold['tn']}"
    )

    print(
        f"FP              : "
        f"{best_threshold['fp']}"
    )

    print(
        f"FN              : "
        f"{best_threshold['fn']}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print()

    print(
        "                 Predicted"
    )

    print(
        "                 Relevant   Irrelevant"
    )

    print(
        f"Actual Relevant   "
        f"{best_threshold['tp']:<11}"
        f"{best_threshold['fn']}"
    )

    print(
        f"Actual Irrelevant "
        f"{best_threshold['fp']:<11}"
        f"{best_threshold['tn']}"
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(
        "TP = relevant question correctly accepted."
    )

    print(
        "TN = irrelevant question correctly rejected."
    )

    print(
        "FP = irrelevant question incorrectly accepted."
    )

    print(
        "FN = relevant question incorrectly rejected."
    )

    print(
        "Precision = percentage of accepted questions "
        "that were actually relevant."
    )

    print(
        "Recall = percentage of relevant questions "
        "that were successfully accepted."
    )

    print(
        "F1 = balance between precision and recall."
    )

    print(
        "Accuracy = percentage of all questions "
        "classified correctly."
    )

    print(
        "Rejection Rate = percentage of irrelevant "
        "questions correctly rejected."
    )

    print()
    print(
        "Use the best threshold as the starting "
        "relevance threshold for the RAG pipeline."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()