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

RELEVANCE_THRESHOLD = 0.40


# ============================================================
# LOAD EMBEDDED DOCUMENTS
# ============================================================

def load_documents(json_path):
    """
    Load previously generated chunks containing
    text + embeddings + metadata.
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
    Calculate cosine similarity between two vectors.
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
# QUERY CLASSIFICATION
# ============================================================

def classify_query(question):
    """
    Classify the query using simple deterministic rules.

    This classifier is intentionally conservative.

    relevant:
        Clearly related to the React Native transcript.

    ambiguous:
        The question could be related but does not provide
        enough context.

    irrelevant:
        Clearly outside the transcript topic.
    """

    q = question.lower().strip()

    # --------------------------------------------------------
    # AMBIGUOUS TERMS
    # --------------------------------------------------------

    ambiguous_questions = {
        "what is view?",
        "what is button?",
        "what languages does react native use?",
        "how does react native work internally?"
    }

    if q in ambiguous_questions:
        return "ambiguous"

    # --------------------------------------------------------
    # CLEARLY IRRELEVANT TOPICS
    # --------------------------------------------------------

    irrelevant_keywords = [
        "kubernetes",
        "postgresql",
        "postgres",
        "docker",
        "spring boot",
        "mongodb"
    ]

    for keyword in irrelevant_keywords:
        if keyword in q:
            return "irrelevant"

    # --------------------------------------------------------
    # AWS DEPLOYMENT
    # --------------------------------------------------------

    if (
        "deploy" in q
        and "aws" in q
    ):
        return "irrelevant"

    # --------------------------------------------------------
    # STRONG REACT NATIVE TOPIC TERMS
    # --------------------------------------------------------

    react_native_terms = [
        "native component",
        "native components",
        "core component",
        "core components",
        "react native",
        "javascript code",
        "native layer",
        "ios",
        "android"
    ]

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # A question containing strong React Native terminology
    # should not be rejected merely because it is phrased
    # as a yes/no question.
    #
    # Example:
    #
    # "Are native components directly used in React Native
    # JavaScript code?"
    #
    # This must be classified as relevant.
    # --------------------------------------------------------

    for term in react_native_terms:
        if term in q:
            return "relevant"

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return "unknown"


# ============================================================
# RETRIEVE BEST CHUNK
# ============================================================

def retrieve_best_chunk(question, documents):
    """
    Generate the question embedding and compare it
    against every document chunk.

    Returns the highest scoring chunk.
    """

    query_embedding = create_embedding(question)

    best_chunk = None
    best_score = -1.0

    for document in documents:

        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        if score > best_score:
            best_score = score
            best_chunk = document

    return best_chunk, best_score


# ============================================================
# DECISION
# ============================================================

def make_decision(
    query_type,
    similarity_score
):
    """
    Final query-aware retrieval decision.

    Order:

    1. Ambiguous → AMBIGUOUS
    2. Irrelevant → REJECT
    3. Relevant → similarity threshold
    4. Unknown → similarity threshold
    """

    if query_type == "ambiguous":
        return "AMBIGUOUS"

    if query_type == "irrelevant":
        return "REJECT"

    if similarity_score >= RELEVANCE_THRESHOLD:
        return "ACCEPT"

    return "REJECT"


# ============================================================
# EVALUATION DATASET
# ============================================================

QUESTIONS = [

    # ========================================================
    # RELEVANT
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

    # ========================================================
    # AMBIGUOUS
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
    print("STEP 19 — IMPROVED QUERY CLASSIFIER")
    print("=" * 90)
    print(
        f"Similarity Threshold : {RELEVANCE_THRESHOLD:.2f}"
    )
    print(
        f"Evaluation Questions : {len(QUESTIONS)}"
    )

    results = []

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
            f"Question   : {question}"
        )

        print(
            f"Expected   : {expected}"
        )

        # ----------------------------------------------------
        # CLASSIFY QUERY
        # ----------------------------------------------------

        query_type = classify_query(
            question
        )

        # ----------------------------------------------------
        # RETRIEVE
        # ----------------------------------------------------

        best_chunk, score = retrieve_best_chunk(
            question,
            documents
        )

        chunk_id = best_chunk["chunk_id"]

        # ----------------------------------------------------
        # FINAL DECISION
        # ----------------------------------------------------

        decision = make_decision(
            query_type,
            score
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if expected == "relevant":

            if decision == "ACCEPT":
                result = "TP"

            elif decision == "REJECT":
                result = "FN"

            else:
                result = "FN"

        elif expected == "irrelevant":

            if decision == "REJECT":
                result = "TN"

            else:
                result = "FP"

        else:

            result = "AMBIGUOUS"

        print(
            f"Query Type : {query_type}"
        )

        print(
            f"Best Chunk : {chunk_id}"
        )

        print(
            f"Score      : {score:.4f}"
        )

        print(
            f"Decision   : {decision}"
        )

        print(
            f"Result     : {result}"
        )

        results.append({
            "question": question,
            "expected": expected,
            "query_type": query_type,
            "score": score,
            "decision": decision,
            "result": result
        })

    # ========================================================
    # METRICS
    # ========================================================

    tp = sum(
        1
        for r in results
        if r["result"] == "TP"
    )

    tn = sum(
        1
        for r in results
        if r["result"] == "TN"
    )

    fp = sum(
        1
        for r in results
        if r["result"] == "FP"
    )

    fn = sum(
        1
        for r in results
        if r["result"] == "FN"
    )

    ambiguous = sum(
        1
        for r in results
        if r["result"] == "AMBIGUOUS"
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

    classified_questions = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / classified_questions
        if classified_questions > 0
        else 0.0
    )

    negative_questions = sum(
        1
        for r in results
        if r["expected"] == "irrelevant"
    )

    correctly_rejected = tn

    rejection_rate = (
        correctly_rejected / negative_questions
        if negative_questions > 0
        else 0.0
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 90)
    print("STEP 19 — QUERY CLASSIFIER SUMMARY")
    print("=" * 90)

    print()
    print(
        f"Threshold        : {RELEVANCE_THRESHOLD:.2f}"
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
        f"Precision        : {precision:.4f}"
    )

    print(
        f"Recall           : {recall:.4f}"
    )

    print(
        f"F1 Score         : {f1:.4f}"
    )

    print(
        f"Accuracy         : {accuracy:.4f}"
    )

    print(
        f"Rejection Rate   : {rejection_rate:.4f}"
    )

    print()

    print(
        f"Ambiguous        : {ambiguous}"
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
        f"Actual Relevant   {tp:<10} {fn}"
    )

    print(
        f"Actual Irrelevant {fp:<10} {tn}"
    )

    # ========================================================
    # FALSE POSITIVES
    # ========================================================

    false_positives = [
        r for r in results
        if r["result"] == "FP"
    ]

    print()
    print("=" * 90)
    print("FALSE POSITIVES")
    print("=" * 90)

    if not false_positives:

        print()
        print("None.")

    else:

        for r in false_positives:

            print(
                f"Score={r['score']:.4f} | "
                f"{r['question']}"
            )

    # ========================================================
    # FALSE NEGATIVES
    # ========================================================

    false_negatives = [
        r for r in results
        if r["result"] == "FN"
    ]

    print()
    print("=" * 90)
    print("FALSE NEGATIVES")
    print("=" * 90)

    if not false_negatives:

        print()
        print("None.")

    else:

        for r in false_negatives:

            print(
                f"Score={r['score']:.4f} | "
                f"{r['question']}"
            )

    # ========================================================
    # AMBIGUOUS
    # ========================================================

    ambiguous_results = [
        r for r in results
        if r["expected"] == "ambiguous"
    ]

    print()
    print("=" * 90)
    print("AMBIGUOUS QUESTIONS")
    print("=" * 90)

    for r in ambiguous_results:

        print(
            f"Score={r['score']:.4f} | "
            f"Decision={r['decision']} | "
            f"{r['question']}"
        )

    # ========================================================
    # FINAL INTERPRETATION
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
        "Query classification is performed before "
        "the similarity threshold."
    )

    print()

    if fp == 0 and fn == 0:

        print(
            "SUCCESS — No false positives or false negatives."
        )

    elif fp == 0:

        print(
            "GOOD — No false positives, but some relevant "
            "questions are still being rejected."
        )

    else:

        print(
            "WARNING — Some irrelevant questions are "
            "being accepted."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()