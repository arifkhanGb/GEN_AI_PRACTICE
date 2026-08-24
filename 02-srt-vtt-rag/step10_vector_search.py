import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

"""
Right now step10_vector_search.py is doing several things:
        Load data
        ↓
        Create query embedding
        ↓
        Search vectors
        ↓
        Print results

     That's fine for learning.   
"""

# This reads: output/embedded_chunks.json

def load_embedded_documents(json_path):
    """
    Load documents that already contain
    text, metadata, and embedding vectors.
    """

    with json_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data["documents"]


# Create an embedding for the user query
def create_embedding(text, api_key):
    
    """
    Convert user query into an embedding vector.
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


# Calculate cosine similarity
def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.

    Result is approximately between:
        -1 and +1

    Higher score = more similar direction.
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

def search_documents(
    query_embedding,
    documents,
    top_k=3
):
    """
    Compare the query vector against
    every stored document vector.

    Returns the top_k most similar documents.
    """

    results = []

    for document in documents:

        document_embedding = document[
            "embedding"
        ]

        score = cosine_similarity(
            query_embedding,
            document_embedding
        )

        results.append({

            "chunk_id": document[
                "chunk_id"
            ],

            "score": score,

            "page_content": document[
                "page_content"
            ],

            "metadata": document[
                "metadata"
            ]
        })


    # -----------------------------------------------------
    # Highest similarity first
    # -----------------------------------------------------

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    return results[:top_k]



def main():

    # =====================================================
    # API key
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
    # Load embedded documents
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
        f"Loaded {len(documents)} embedded documents."
    )


    # =====================================================
    # User query
    # =====================================================

    query = (
        "What are native components?"
    )


    print("\n")
    print("=" * 60)
    print("USER QUERY")
    print("=" * 60)

    print(query)


    # =====================================================
    # Convert query → embedding
    # =====================================================

    print("\nCreating query embedding...")

    query_embedding = create_embedding(
        query,
        api_key
    )


    print(
        "Query vector dimensions:",
        len(query_embedding)
    )


    # =====================================================
    # Search
    # =====================================================

    results = search_documents(

        query_embedding=query_embedding,

        documents=documents,

        top_k=3
    )


    # =====================================================
    # Display results
    # =====================================================

    print("\n")
    print("=" * 60)
    print("TOP RESULTS")
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
            "-" * 60
        )

        print(
            f"Chunk ID: {result['chunk_id']}"
        )

        print(
            f"Similarity: {result['score']:.4f}"
        )

        print(
            f"Timestamp: "
            f"{result['metadata']['start']} "
            f"--> "
            f"{result['metadata']['end']}"
        )

        print("\nText:")

        print(
            result["page_content"]
        )


if __name__ == "__main__":
    main()
