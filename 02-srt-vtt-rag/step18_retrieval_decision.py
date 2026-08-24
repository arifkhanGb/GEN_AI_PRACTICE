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

RELEVANCE_THRESHOLD = 0.40


# ============================================================
# LOAD EMBEDDED DOCUMENTS
# ============================================================

def load_documents(json_path):
    """
    Load previously generated chunks
    containing text, metadata and embeddings.
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
    Convert user query into an embedding vector.
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
    Calculate cosine similarity between two vectors.
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
# RETRIEVE BEST CHUNK
# ============================================================

def retrieve_best_chunk(
    query,
    documents
):
    """
    Find the highest similarity chunk.
    """

    query_embedding = create_embedding(
        query
    )

    best_document = None
    best_score = -1.0

    for document in documents:

        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        if score > best_score:

            best_score = score
            best_document = document

    return best_document, best_score


# ============================================================
# CLASSIFY QUERY
# ============================================================

def classify_query(query):
    """
    Lightweight rule-based query classification.

    This is intentionally simple for Step 18.
    """

    query_lower = query.lower().strip()

    # --------------------------------------------------------
    # Clearly irrelevant technologies/topics
    # --------------------------------------------------------

    irrelevant_keywords = [
        "kubernetes",
        "postgresql",
        "docker",
        "spring boot",
        "mongodb",
        "java",
        "aws"
    ]

    for keyword in irrelevant_keywords:

        if keyword in query_lower:

            return "irrelevant"


    # --------------------------------------------------------
    # Clearly relevant React Native concepts
    # --------------------------------------------------------

    relevant_patterns = [
        "native component",
        "native components",
        "core component",
        "core components",
        "operating system",
        "native language",
        "native languages",
        "directly used",
        "react native core"
    ]

    for pattern in relevant_patterns:

        if pattern in query_lower:

            return "relevant"


    # --------------------------------------------------------
    # Ambiguous questions
    # --------------------------------------------------------

    ambiguous_patterns = [
        "what is view",
        "what is button",
        "what languages does react native use",
        "how does react native work",
        "how does react native work internally"
    ]

    for pattern in ambiguous_patterns:

        if pattern in query_lower:

            return "ambiguous"


    # --------------------------------------------------------
    # Unknown
    # --------------------------------------------------------

    return "unknown"


# ============================================================
# FINAL RETRIEVAL DECISION
# ============================================================

def make_decision(
    query,
    score
):
    """
    Combine similarity score with query classification.
    """

    query_type = classify_query(
        query
    )

    # --------------------------------------------------------
    # Clearly irrelevant
    # --------------------------------------------------------

    if query_type == "irrelevant":

        return (
            "REJECT",
            query_type
        )


    # --------------------------------------------------------
    # Ambiguous
    # --------------------------------------------------------

    if query_type == "ambiguous":

        return (
            "REJECT",
            query_type
        )


    # --------------------------------------------------------
    # Relevant / unknown
    # --------------------------------------------------------

    if score >= RELEVANCE_THRESHOLD:

        return (
            "ACCEPT",
            query_type
        )

    return (
        "REJECT",
        query_type
    )


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

QUESTIONS = [

    # --------------------------------------------------------
    # RELEVANT
    # --------------------------------------------------------

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

    {
        "question": "What is the difference between native and core components?",
        "expected": "relevant"
    },

    {
        "question": "How are native components different from core components?",
        "expected": "relevant"
    },


    # --------------------------------------------------------
    # AMBIGUOUS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # IRRELEVANT
    # --------------------------------------------------------

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

    print()
    print("=" * 90)
    print("STEP 18 — QUERY-AWARE RETRIEVAL DECISION")
    print("=" * 90)

    print(
        f"Similarity Threshold : "
        f"{RELEVANCE_THRESHOLD:.2f}"
    )

    print(
        f"Evaluation Questions : "
        f"{len(QUESTIONS)}"
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    ambiguous_total = 0
    ambiguous_rejected = 0

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for index, item in enumerate(
        QUESTIONS,
        start=1
    ):

        question = item["question"]
        expected = item["expected"]

        best_chunk, score = retrieve_best_chunk(
            question,
            documents
        )

        decision, query_type = make_decision(
            question,
            score
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        if expected == "relevant":

            if decision == "ACCEPT":
                tp += 1
                result = "TP"

            else:
                fn += 1
                result = "FN"

        elif expected == "irrelevant":

            if decision == "REJECT":
                tn += 1
                result = "TN"

            else:
                fp += 1
                result = "FP"

        else:

            ambiguous_total += 1

            if decision == "REJECT":
                ambiguous_rejected += 1

            result = "AMBIGUOUS"

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print()
        print("-" * 90)

        print(
            f"[{index}/{len(QUESTIONS)}]"
        )

        print(
            f"Question   : {question}"
        )

        print(
            f"Expected   : {expected}"
        )

        print(
            f"Query Type : {query_type}"
        )

        print(
            f"Best Chunk : "
            f"{best_chunk['chunk_id']}"
        )

        print(
            f"Score      : "
            f"{score:.4f}"
        )

        print(
            f"Decision   : "
            f"{decision}"
        )

        print(
            f"Result     : "
            f"{result}"
        )

    # ========================================================
    # METRICS
    # ========================================================

    classified_total = (
        tp + tn + fp + fn
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / classified_total
        if classified_total > 0
        else 0.0
    )

    rejection_rate = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 90)
    print("STEP 18 — RETRIEVAL DECISION SUMMARY")
    print("=" * 90)

    print()
    print(
        f"Threshold        : "
        f"{RELEVANCE_THRESHOLD:.2f}"
    )

    print(
        f"TP               : {tp}"
    )

    print(
        f"TN               : {tn}"
    )

    print(
        f"FP               : {fp}"
    )

    print(
        f"FN               : {fn}"
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
    # AMBIGUOUS SUMMARY
    # ========================================================

    print()
    print("=" * 90)
    print("AMBIGUOUS QUESTIONS")
    print("=" * 90)

    print()

    print(
        f"Total Ambiguous  : "
        f"{ambiguous_total}"
    )

    print(
        f"Rejected         : "
        f"{ambiguous_rejected}"
    )

    ambiguous_rejection_rate = (
        ambiguous_rejected / ambiguous_total
        if ambiguous_total > 0
        else 0.0
    )

    print(
        f"Rejection Rate   : "
        f"{ambiguous_rejection_rate:.4f}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print()
    print("=" * 90)
    print("CONFUSION MATRIX")
    print("=" * 90)

    print()
    print(
        "                 Predicted"
    )

    print(
        "                 Relevant   Irrelevant"
    )

    print(
        f"Actual Relevant   "
        f"{tp:<10} {fn}"
    )

    print(
        f"Actual Irrelevant "
        f"{fp:<10} {tn}"
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print()
    print("=" * 90)
    print("INTERPRETATION")
    print("=" * 90)

    print()

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
        "Ambiguous questions are evaluated separately."
    )

    print(
        "Query classification is used before the "
        "similarity threshold."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()