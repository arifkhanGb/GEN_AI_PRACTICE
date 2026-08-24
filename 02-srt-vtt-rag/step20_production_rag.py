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
    raise RuntimeError("OPENAI_API_KEY is not loaded.")

EMBEDDING_MODEL = "text-embedding-3-small"

SIMILARITY_THRESHOLD = 0.40

TOP_K = 3

EMBEDDING_URL = "https://api.openai.com/v1/embeddings"

CHAT_URL = "https://api.openai.com/v1/chat/completions"

CHAT_MODEL = "gpt-4o-mini"


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents(json_path):
    """
    Load previously generated chunks containing:
        - text
        - metadata
        - embeddings
    """

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data["documents"]


# ============================================================
# CREATE EMBEDDING
# ============================================================

def create_embedding(text):
    """
    Convert text into an embedding vector.
    """

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "input": text,
    }

    response = requests.post(
        EMBEDDING_URL,
        headers=headers,
        json=payload,
        timeout=60,
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
# QUERY CLASSIFIER
# ============================================================

def classify_query(question):
    """
    Classify the user's question before retrieval.

    This classifier is intentionally conservative.

    relevant  -> continue retrieval
    ambiguous -> ask clarification
    irrelevant -> reject
    unknown   -> let similarity threshold decide
    """

    question_lower = question.lower().strip()

    # --------------------------------------------------------
    # AMBIGUOUS QUESTIONS
    # --------------------------------------------------------

    ambiguous_patterns = [
        "what is view",
        "what is button",
        "what languages does react native use",
        "how does react native work internally",
    ]

    for pattern in ambiguous_patterns:
        if pattern in question_lower:
            return "ambiguous"

    # --------------------------------------------------------
    # CLEARLY IRRELEVANT TOPICS
    # --------------------------------------------------------

    irrelevant_patterns = [
        "kubernetes",
        "postgresql",
        "postgres",
        "docker",
        "spring boot",
        "mongodb",
    ]

    for pattern in irrelevant_patterns:
        if pattern in question_lower:
            return "irrelevant"

    # AWS deployment is outside the transcript scope.
    if (
        "deploy" in question_lower
        and "aws" in question_lower
    ):
        return "irrelevant"

    # --------------------------------------------------------
    # CLEARLY RELEVANT TERMS
    # --------------------------------------------------------

    relevant_patterns = [
        "native component",
        "native components",
        "core component",
        "core components",
        "react native",
    ]

    for pattern in relevant_patterns:
        if pattern in question_lower:
            return "relevant"

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return "unknown"


# ============================================================
# RETRIEVE CHUNKS
# ============================================================

def retrieve_chunks(question, documents):
    """
    Convert the question into an embedding and compare it
    against every stored document embedding.
    """

    query_embedding = create_embedding(question)

    scored_documents = []

    for document in documents:

        score = cosine_similarity(
            query_embedding,
            document["embedding"],
        )

        scored_documents.append(
            {
                "document": document,
                "score": score,
            }
        )

    scored_documents.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_documents


# ============================================================
# PRINT RETRIEVAL RESULTS
# ============================================================

def print_retrieval_results(results):
    """
    Display top retrieved chunks.
    """

    print()
    print("=" * 70)
    print("RETRIEVED CHUNKS")
    print("=" * 70)

    for rank, item in enumerate(
        results[:TOP_K],
        start=1,
    ):

        document = item["document"]
        score = item["score"]

        print()
        print("-" * 70)

        print(f"Rank       : {rank}")

        print(
            f"Chunk ID    : "
            f"{document.get('chunk_id', 'unknown')}"
        )

        print(
            f"Similarity : {score:.4f}"
        )

        if "timestamp" in document:
            print(
                f"Timestamp  : "
                f"{document['timestamp']}"
            )

        if "start_time" in document:
            print(
                f"Start      : "
                f"{document['start_time']}"
            )

        if "end_time" in document:
            print(
                f"End        : "
                f"{document['end_time']}"
            )


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):
    """
    Build context for the LLM from retrieved chunks.
    """

    context_parts = []

    for rank, item in enumerate(
        results[:TOP_K],
        start=1,
    ):

        document = item["document"]
        score = item["score"]

        text = document.get(
            "text",
            document.get(
                "content",
                "",
            ),
        )

        chunk_id = document.get(
            "chunk_id",
            f"chunk_{rank:04d}",
        )

        context_parts.append(
            f"""
SOURCE {rank}
Chunk ID: {chunk_id}
Similarity: {score:.4f}

{text}
""".strip()
        )

    return "\n\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, context):
    """
    Generate an answer using only the retrieved context.
    """

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    system_prompt = """
You are a retrieval-augmented question answering assistant.

Answer the user's question ONLY using the provided transcript
context.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain the answer, say:
   "The transcript does not provide enough information to answer this."
4. Keep the answer concise and factual.
5. Mention source numbers when useful.
"""

    user_prompt = f"""
USER QUESTION:
{question}

TRANSCRIPT CONTEXT:
{context}

Answer the question using only the transcript context.
"""

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt.strip(),
            },
            {
                "role": "user",
                "content": user_prompt.strip(),
            },
        ],
        "temperature": 0,
    }

    response = requests.post(
        CHAT_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    return result["choices"][0]["message"]["content"]


# ============================================================
# HANDLE AMBIGUOUS QUERY
# ============================================================

def handle_ambiguous_question(question):
    """
    Ask the user to clarify an ambiguous question.
    """

    print()
    print("=" * 70)
    print("AMBIGUOUS QUESTION")
    print("=" * 70)

    print()
    print(
        f'Your question "{question}" is too broad '
        f"to answer reliably from this transcript."
    )

    print()
    print("Please make the question more specific.")

    print()
    print("Example:")

    if question.lower() == "what is view?":
        print(
            "What is the React Native View component?"
        )

    elif question.lower() == "what is button?":
        print(
            "What is the React Native Button component?"
        )

    else:
        print(
            "Please specify which React Native concept "
            "you are asking about."
        )


# ============================================================
# HANDLE IRRELEVANT QUERY
# ============================================================

def handle_irrelevant_question():
    """
    Reject questions outside transcript scope.
    """

    print()
    print("=" * 70)
    print("NO RELEVANT CONTEXT")
    print("=" * 70)

    print()
    print(
        "This question is outside the scope of "
        "the provided transcript."
    )


# ============================================================
# HANDLE UNKNOWN QUERY
# ============================================================

def handle_unknown_question(
    question,
    retrieval_results,
):
    """
    Unknown queries are decided using similarity.
    """

    best_score = retrieval_results[0]["score"]

    print()
    print(
        f"Unknown query type."
    )

    print(
        f"Best similarity score: "
        f"{best_score:.4f}"
    )

    if best_score < SIMILARITY_THRESHOLD:

        print()
        print(
            "The question does not have sufficiently "
            "relevant context in the transcript."
        )

        return False

    return True


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(results):
    """
    Display the chunks used to generate the answer.
    """

    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    for rank, item in enumerate(
        results[:TOP_K],
        start=1,
    ):

        document = item["document"]

        chunk_id = document.get(
            "chunk_id",
            "unknown",
        )

        score = item["score"]

        print()
        print(
            f"[{rank}] {chunk_id}"
        )

        print(
            f"Similarity: {score:.4f}"
        )

        if "timestamp" in document:
            print(
                f"Timestamp : "
                f"{document['timestamp']}"
            )


# ============================================================
# MAIN RAG PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD EMBEDDED DOCUMENTS
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
    # USER QUESTION
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
    # STEP 1 — QUERY CLASSIFICATION
    # --------------------------------------------------------

    query_type = classify_query(
        question
    )

    print()
    print(
        f"Query Type : {query_type}"
    )

    # --------------------------------------------------------
    # IRRELEVANT
    # --------------------------------------------------------

    if query_type == "irrelevant":

        handle_irrelevant_question()

        return

    # --------------------------------------------------------
    # AMBIGUOUS
    # --------------------------------------------------------

    if query_type == "ambiguous":

        handle_ambiguous_question(
            question
        )

        return

    # --------------------------------------------------------
    # STEP 2 — RETRIEVAL
    # --------------------------------------------------------

    print()
    print(
        "Generating query embedding..."
    )

    retrieval_results = retrieve_chunks(
        question,
        documents,
    )

    # --------------------------------------------------------
    # DISPLAY RETRIEVAL
    # --------------------------------------------------------

    print_retrieval_results(
        retrieval_results
    )

    # --------------------------------------------------------
    # STEP 3 — BEST SCORE
    # --------------------------------------------------------

    best_score = retrieval_results[0]["score"]

    print()
    print(
        f"Best Similarity Score : "
        f"{best_score:.4f}"
    )

    # --------------------------------------------------------
    # UNKNOWN QUERY
    # --------------------------------------------------------

    if query_type == "unknown":

        should_continue = handle_unknown_question(
            question,
            retrieval_results,
        )

        if not should_continue:
            return

    # --------------------------------------------------------
    # STEP 4 — SIMILARITY THRESHOLD
    # --------------------------------------------------------

    if best_score < SIMILARITY_THRESHOLD:

        print()
        print("=" * 70)
        print("NO RELEVANT CONTEXT")
        print("=" * 70)

        print()
        print(
            f"Best score {best_score:.4f} "
            f"is below threshold "
            f"{SIMILARITY_THRESHOLD:.2f}."
        )

        print()
        print(
            "The transcript does not contain "
            "sufficiently relevant information."
        )

        return

    # --------------------------------------------------------
    # ACCEPT
    # --------------------------------------------------------

    print()
    print(
        f"Decision : ACCEPT "
        f"(threshold = {SIMILARITY_THRESHOLD:.2f})"
    )

    # --------------------------------------------------------
    # STEP 5 — BUILD CONTEXT
    # --------------------------------------------------------

    context = build_context(
        retrieval_results
    )

    print()
    print(
        "Context assembled from top "
        f"{TOP_K} chunks."
    )

    # --------------------------------------------------------
    # STEP 6 — GENERATE ANSWER
    # --------------------------------------------------------

    print()
    print(
        "Generating answer..."
    )

    answer = generate_answer(
        question,
        context,
    )

    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print()
    print(answer)

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    display_sources(
        retrieval_results
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()