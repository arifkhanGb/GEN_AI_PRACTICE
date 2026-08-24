import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

def load_embedded_documents(json_path):
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


def create_embedding(text, api_key):
    """
    Convert the user's question into
    the same vector space used by
    our document embeddings.
    """

    url = "https://api.openai.com/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {api_key}",
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



def retrieve(
    query,
    documents,
    api_key,
    top_k=3,
    min_score=0.0
):
    """
    Retrieve the most relevant documents
    for a user query.

    Parameters
    ----------
    query:
        User's natural-language question.

    documents:
        Documents with precomputed embeddings.

    api_key:
        API key used to embed the query.

    top_k:
        Maximum number of chunks to return.

    min_score:
        Minimum similarity score allowed.
    """

    # -----------------------------------------------------
    # Step 1: Convert query into a vector
    # -----------------------------------------------------

    query_embedding = create_embedding(
        query,
        api_key
    )


    # -----------------------------------------------------
    # Step 2: Compare query against every document
    # -----------------------------------------------------

    results = []

    for document in documents:

        score = cosine_similarity(

            query_embedding,

            document["embedding"]
        )


        # -------------------------------------------------
        # Keep only results above the minimum threshold
        # -------------------------------------------------

        if score >= min_score:

            results.append({

                "chunk_id": document[
                    "chunk_id"
                ],

                "score": score,

                "text": document[
                    "page_content"
                ],

                "metadata": document[
                    "metadata"
                ]
            })


    # -----------------------------------------------------
    # Step 3: Sort highest similarity first
    # -----------------------------------------------------

    results.sort(

        key=lambda item: item["score"],

        reverse=True
    )


    # -----------------------------------------------------
    # Step 4: Return Top-K
    # -----------------------------------------------------

    return results[:top_k]

def format_context(results):
    """
    Convert retrieved chunks into a clean
    text block that can later be sent to an LLM.
    """

    if not results:

        return (
            "No relevant context was found."
        )


    context_parts = []


    for index, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]


        context = f"""
SOURCE {index}
------------------------------
Chunk ID: {result["chunk_id"]}
Similarity: {result["score"]:.4f}

Source File:
{metadata["source_file"]}

Timestamp:
{metadata["start"]} --> {metadata["end"]}

Text:
{result["text"]}
"""


        context_parts.append(
            context.strip()
        )


    return "\n\n".join(
        context_parts
    )

def format_context(results):
    """
    Convert retrieved chunks into a clean
    text block that can later be sent to an LLM.
    """

    if not results:

        return (
            "No relevant context was found."
        )


    context_parts = []


    for index, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]


        context = f"""
SOURCE {index}
------------------------------
Chunk ID: {result["chunk_id"]}
Similarity: {result["score"]:.4f}

Source File:
{metadata["source_file"]}

Timestamp:
{metadata["start"]} --> {metadata["end"]}

Text:
{result["text"]}
"""


        context_parts.append(
            context.strip()
        )


    return "\n\n".join(
        context_parts
    )



def main():

    # =====================================================
    # Load API key
    # =====================================================

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        print(
            "ERROR: OPENAI_API_KEY is not loaded."
        )

        return


    # =====================================================
    # Load embedded chunks
    # =====================================================

    input_path = Path(
        "output/embedded_chunks.json"
    )


    if not input_path.exists():

        print(
            f"File not found: {input_path}"
        )

        return


    documents = load_embedded_documents(
        input_path
    )


    print(
        f"Loaded {len(documents)} documents."
    )


    # =====================================================
    # User question
    # =====================================================

    query = (
        "What are native components?"
    )


    print("\n")
    print("=" * 60)
    print("QUERY")
    print("=" * 60)

    print(query)


    # =====================================================
    # Retrieve relevant chunks
    # =====================================================

    results = retrieve(

        query=query,

        documents=documents,

        api_key=api_key,

        top_k=3,

        min_score=0.0
    )


    # =====================================================
    # Display retrieved results
    # =====================================================

    print("\n")
    print("=" * 60)
    print("RETRIEVED RESULTS")
    print("=" * 60)


    for rank, result in enumerate(
        results,
        start=1
    ):

        print("\n")

        print(
            f"RANK {rank}"
        )

        print(
            f"Chunk: {result['chunk_id']}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Timestamp: "
            f"{result['metadata']['start']} "
            f"--> "
            f"{result['metadata']['end']}"
        )

        print("\nText:")

        print(
            result["text"]
        )


    # =====================================================
    # Build LLM-ready context
    # =====================================================

    context = format_context(
        results
    )


    print("\n")
    print("=" * 60)
    print("LLM-READY CONTEXT")
    print("=" * 60)

    print(context)


if __name__ == "__main__":
    main()


