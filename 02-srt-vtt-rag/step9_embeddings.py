import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

def create_embedding(text, api_key):
    """
    Send text to the embedding API
    and return the embedding vector.
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

    # Raise an exception if the API
    # returned an HTTP error.
    response.raise_for_status()

    result = response.json()

    embedding = result["data"][0]["embedding"]

    return embedding



def load_documents(json_path):
    """
    Load chunks from chunks.json
    and convert them into document structures.
    """

    with json_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    documents = []

    for chunk in data["chunks"]:

        document = {

            "page_content": chunk["text"],

            "metadata": {

                "chunk_id": chunk["chunk_id"],

                "source_file": chunk["source_file"],

                "start": chunk["start"],

                "end": chunk["end"],

                "start_seconds": chunk["start_seconds"],

                "end_seconds": chunk["end_seconds"],

                "subtitle_start": chunk["subtitle_start"],

                "subtitle_end": chunk["subtitle_end"],

                "token_count": chunk["token_count"]
            }
        }

        documents.append(
            document
        )

    return documents

def embed_documents(documents, api_key):
    """
    Create an embedding for every document.

    Each document keeps:
        - page_content
        - metadata
        - embedding

    The original document is not modified.
    """

    embedded_documents = []

    total_documents = len(documents)

    for index, document in enumerate(
        documents,
        start=1
    ):

        chunk_id = document["metadata"]["chunk_id"]

        print(
            f"Embedding {index}/{total_documents}: {chunk_id}"
        )

        # -------------------------------------------------
        # Send document text to embedding model
        # -------------------------------------------------

        embedding = create_embedding(
            document["page_content"],
            api_key
        )

        # -------------------------------------------------
        # Create a new object containing:
        #
        # text
        # metadata
        # vector
        # -------------------------------------------------

        embedded_document = {

            "chunk_id": chunk_id,

            "page_content": document[
                "page_content"
            ],

            "metadata": document[
                "metadata"
            ],

            "embedding": embedding
        }

        embedded_documents.append(
            embedded_document
        )

    return embedded_documents

    # JSON-saving function

def save_embedded_documents(
    documents,
    output_path
):
    """
    Save documents + embeddings to JSON.
    """

    output_data = {

        "total_documents": len(documents),

        "embedding_dimensions": (
            len(documents[0]["embedding"])
            if documents
            else 0
        ),

        "documents": documents
    }


    # -----------------------------------------------------
    # Create output directory if necessary
    # -----------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            indent=2,
            ensure_ascii=False
        )


    print(
        f"\nEmbedded documents saved to: "
        f"{output_path}"
    )       


def main():

    # -----------------------------------------------------
    # Load API key
    # -----------------------------------------------------

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        print(
            "ERROR: OPENAI_API_KEY is not set."
        )

        return


    # -----------------------------------------------------
    # JSON containing our chunks
    # -----------------------------------------------------

    input_path = Path(
        "output/chunks.json"
    )


    if not input_path.exists():

        print(
            f"File not found: {input_path}"
        )

        return


    # -----------------------------------------------------
    # Load documents
    # -----------------------------------------------------

    documents = load_documents(
        input_path
    )


    print(
        f"Documents loaded: {len(documents)}"
    )

     # =====================================================
     # Create embeddings for ALL documents
     # =====================================================

    embedded_documents = embed_documents(

        documents=documents,

        api_key=api_key
    )


    # =====================================================
    # Save embeddings
    # =====================================================

    output_path = Path(
        "output/embedded_chunks.json"
    )


    save_embedded_documents(

        documents=embedded_documents,

        output_path=output_path
    )



    # =====================================================
    # Basic verification
    # =====================================================

    print("\n")
    print("=" * 60)
    print("EMBEDDING VERIFICATION")
    print("=" * 60)


    for document in embedded_documents:

        print(
            document["chunk_id"],
            "→",
            len(document["embedding"]),
            "dimensions"
        )



    # # -----------------------------------------------------
    # # Test with ONE document
    # # -----------------------------------------------------

    # document = documents[0]


    # print(
    #     "\nCreating embedding for:"
    # )

    # print(
    #     document["metadata"]["chunk_id"]
    # )


    # # -----------------------------------------------------
    # # Create embedding
    # # -----------------------------------------------------

    # embedding = create_embedding(

    #     document["page_content"],

    #     api_key
    # )


    # # -----------------------------------------------------
    # # Inspect result
    # # -----------------------------------------------------

    # print(
    #     "\nEmbedding created successfully."
    # )


    # print(
    #     "Vector dimensions:",
    #     len(embedding)
    # )


    # print(
    #     "\nFirst 10 values:"
    # )


    # print(
    #     embedding[:10]    
    # )


if __name__ == "__main__":
    main()    
