import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

COLLECTION_NAME = "srt_transcript"

INPUT_PATH = Path(
    "output/embedded_chunks.json"
)


# ============================================================
# LOAD EMBEDDED DOCUMENTS
# ============================================================

def load_documents(json_path):
    """
    Load previously generated documents.

    Each document should contain:
        - chunk_id
        - text
        - embedding
        - metadata
    """

    if not json_path.exists():
        raise FileNotFoundError(
            f"File not found: {json_path}"
        )

    with json_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    documents = data["documents"]

    print(
        f"Loaded {len(documents)} documents."
    )

    return documents


# ============================================================
# DETECT VECTOR SIZE
# ============================================================

def get_vector_size(documents):
    """
    Determine embedding dimension from
    the first document.
    """

    if not documents:
        raise ValueError(
            "No documents found."
        )

    embedding = documents[0].get("embedding")

    if not embedding:
        raise ValueError(
            "First document does not contain an embedding."
        )

    return len(embedding)


# ============================================================
# CHECK QDRANT
# ============================================================

def check_qdrant():
    """
    Verify that Qdrant is running.
    """

    url = QDRANT_URL

    response = requests.get(
        url,
        timeout=10
    )

    response.raise_for_status()

    print()
    print("=" * 70)
    print("QDRANT CONNECTION")
    print("=" * 70)

    print(
        f"Qdrant is running: {QDRANT_URL}"
    )


# ============================================================
# DELETE EXISTING COLLECTION
# ============================================================

def delete_collection():
    """
    Delete the collection if it already exists.

    This makes Step 22 repeatable during development.
    """

    url = (
        f"{QDRANT_URL}/collections/"
        f"{COLLECTION_NAME}"
    )

    response = requests.delete(
        url,
        timeout=30
    )

    if response.status_code == 200:
        print(
            f"Existing collection deleted: "
            f"{COLLECTION_NAME}"
        )

    elif response.status_code == 404:
        print(
            "Collection does not exist yet."
        )

    else:
        response.raise_for_status()


# ============================================================
# CREATE COLLECTION
# ============================================================

def create_collection(vector_size):
    """
    Create a Qdrant collection using
    cosine similarity.
    """

    url = (
        f"{QDRANT_URL}/collections/"
        f"{COLLECTION_NAME}"
    )

    payload = {
        "vectors": {
            "size": vector_size,
            "distance": "Cosine"
        }
    }

    response = requests.put(
        url,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    print()
    print("=" * 70)
    print("COLLECTION CREATED")
    print("=" * 70)

    print(
        f"Collection : {COLLECTION_NAME}"
    )

    print(
        f"Vector size: {vector_size}"
    )

    print(
        "Distance   : Cosine"
    )


# ============================================================
# PREPARE PAYLOAD
# ============================================================

# ============================================================
# PREPARE PAYLOAD
# ============================================================

def create_points(documents):
    """
    Convert embedded documents into Qdrant points.

    Input document structure:

        {
            "chunk_id": "...",
            "text": "...",
            "metadata": {...},
            "embedding": [...]
        }

    Qdrant payload structure:

        {
            "chunk_id": "...",
            "text": "...",
            "metadata": {...}
        }
    """

    points = []

    for index, document in enumerate(documents, start=1):

        embedding = document["embedding"]

        # IMPORTANT:
        # embedded_chunks.json uses "text"
        # NOT "text"

        text = document.get("text", "")

        payload = {
            "chunk_id": document.get(
                "chunk_id"
            ),
            "text": document.get(
                "page_content",
                ""
            ),
            "metadata": document.get(
                "metadata",
                {}
            )
        }

        # IMPORTANT VALIDATION
        if not payload["text"]:
            raise ValueError(
                f"Empty text found for {payload['chunk_id']}"
            )

        point = {
            "id": index,
            "vector": embedding,
            "payload": payload
        }

        points.append(point)

    return points

# ============================================================
# INSERT POINTS
# ============================================================

def insert_points(points):
    """
    Insert all embedded chunks into Qdrant.
    """

    url = (
        f"{QDRANT_URL}/collections/"
        f"{COLLECTION_NAME}/points"
    )

    payload = {
        "points": points
    }

    response = requests.put(
        url,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    print()
    print("=" * 70)
    print("VECTORS STORED")
    print("=" * 70)

    print(
        f"Points inserted: {len(points)}"
    )

    print(
        f"Qdrant response: {result}"
    )


# ============================================================
# VERIFY COLLECTION
# ============================================================

def verify_collection():
    """
    Verify that Qdrant contains the
    expected number of vectors.
    """

    url = (
        f"{QDRANT_URL}/collections/"
        f"{COLLECTION_NAME}"
    )

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    result = data["result"]

    points_count = result.get(
        "points_count",
        0
    )

    vectors_count = result.get(
        "vectors_count",
        0
    )

    print()
    print("=" * 70)
    print("QDRANT VERIFICATION")
    print("=" * 70)

    print(
        f"Collection   : {COLLECTION_NAME}"
    )

    print(
        f"Points count : {points_count}"
    )

    print(
        f"Vectors count: {vectors_count}"
    )

    print(
        f"Status       : {result.get('status')}"
    )

    return points_count


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STEP 22 — STORE EMBEDDINGS IN QDRANT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    documents = load_documents(
        INPUT_PATH
    )

    # --------------------------------------------------------
    # Detect vector dimension
    # --------------------------------------------------------

    vector_size = get_vector_size(
        documents
    )

    print(
        f"Embedding dimension: {vector_size}"
    )

    # --------------------------------------------------------
    # Check Qdrant
    # --------------------------------------------------------

    check_qdrant()

    # --------------------------------------------------------
    # Remove old collection
    # --------------------------------------------------------

    delete_collection()

    # --------------------------------------------------------
    # Create collection
    # --------------------------------------------------------

    create_collection(
        vector_size
    )

    # --------------------------------------------------------
    # Convert documents → Qdrant points
    # --------------------------------------------------------

    points = create_points(
        documents
    )

    print()
    print(
        f"Prepared {len(points)} Qdrant points."
    )

    # --------------------------------------------------------
    # Store vectors
    # --------------------------------------------------------

    insert_points(
        points
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    count = verify_collection()

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 22 COMPLETE")
    print("=" * 70)

    if count == len(documents):

        print(
            "SUCCESS — All embedded chunks "
            "are stored in Qdrant."
        )

    else:

        print(
            "WARNING — Stored point count "
            "does not match document count."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()