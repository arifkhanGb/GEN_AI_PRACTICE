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
    Convert text into an embedding vector.
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
    min_score=0.50
):
    """
    Retrieve the most relevant chunks
    for the user's question.

    Process:

        1. Convert query into embedding
        2. Compare query against every document
        3. Apply similarity threshold
        4. Sort by similarity
        5. Return Top-K results
    """

    # =====================================================
    # STEP 1 — Convert query into embedding
    # =====================================================

    query_embedding = create_embedding(
        query,
        api_key
    )

    results = []

    # =====================================================
    # STEP 2 — Compare query against every document
    # =====================================================

    for document in documents:

        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        # -------------------------------------------------
        # DEBUG
        # -------------------------------------------------

        print(
            f"DEBUG → "
            f"{document['chunk_id']} "
            f"= {score:.4f}"
        )

        # =================================================
        # STEP 3 — Apply similarity threshold
        # =================================================

        if score >= min_score:

            results.append({

                "chunk_id": document["chunk_id"],

                "score": score,

                "text": document["page_content"],

                "metadata": document["metadata"]
            })

    # =====================================================
    # STEP 4 — Sort highest similarity first
    # =====================================================

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # =====================================================
    # STEP 5 — Return Top-K
    # =====================================================

    return results[:top_k]



def build_context(results):
    """
    Build structured context for the LLM.

    Each retrieved chunk contains:
    - chunk ID
    - similarity score
    - timestamp
    - subtitle range
    - transcript text

    Keeping this metadata allows us to trace
    the final answer back to the original SRT.
    """

    context_parts = []

    for index, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        context = f"""

SOURCE {index}

Chunk ID:
{result["chunk_id"]}

Similarity Score:
{result["score"]:.4f}

Timestamp:
{metadata["start"]} --> {metadata["end"]}

Subtitle Range:
{metadata["subtitle_start"]} --> {metadata["subtitle_end"]}

Transcript:
{result["text"]}
"""

        context_parts.append(
            context.strip()
        )

    return "\n\n".join(
        context_parts
    )

def display_sources(results):
    """
    Display the transcript sources used for the answer.

    This allows us to trace the generated answer
    back to the original SRT timestamps.
    """

    print("\n")
    print("=" * 60)
    print("SOURCES")
    print("=" * 60)

    for index, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        print(
            f"\n[{index}] "
            f"{result['chunk_id']}"
        )

        print(
            f"Timestamp: "
            f"{metadata['start']} --> "
            f"{metadata['end']}"
        )

        print(
            f"Similarity: "
            f"{result['score']:.4f}"
        )


def build_prompt(query, context):
    """
    Build a strict RAG prompt.

    The LLM must answer only from the retrieved
    transcript and cite the source number.
    """

    prompt = f"""
You are a RAG question-answering assistant.

Answer the user's question using ONLY the transcript
sources provided below.

RULES:

1. Do not use outside knowledge.

2. Do not invent information.

3. Every factual statement in your answer MUST have
   at least one citation.

4. Citation format MUST be exactly:

   [Source 1]

   [Source 2]

   [Source 3]

5. Only use source numbers that actually exist.

6. Put the citation immediately after the statement
   it supports.

7. If multiple sources support a statement, use:

   [Source 1][Source 2]

8. If the answer cannot be found in the transcript,
   respond exactly:

   I don't have enough information in the provided transcript.

9. Keep the answer concise.

10. Do not create a SOURCES section.
    The Python program handles that separately.

==================================================
RETRIEVED TRANSCRIPT
==================================================

{context}

==================================================
USER QUESTION
==================================================

{query}

==================================================
ANSWER
==================================================

Remember: every factual statement requires [Source N].
"""

    return prompt


def generate_answer(
    prompt,
    api_key
):
    """
    Send the RAG prompt to the LLM
    and return the generated answer.
    """

    url = (
        "https://api.openai.com/v1/chat/completions"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {

        "model": "gpt-4o-mini",

        "messages": [

            {
                "role": "system",
                "content": (
                    "You answer questions using "
                    "the supplied context."
                )
            },

            {
                "role": "user",
                "content": prompt
            }
        ],

        "temperature": 0
    }


    response = requests.post(

        url,

        headers=headers,

        json=payload,

        timeout=60
    )


    response.raise_for_status()


    result = response.json()


    return result[
        "choices"
    ][0][
        "message"
    ][
        "content"
    ]

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
        f"Loaded {len(documents)} documents."
    )


    # =====================================================
    # User question
    # =====================================================

    query = input(
        "\nAsk a question about the transcript: "
    ).strip()


    if not query:

        print(
            "Question cannot be empty."
        )

        return


    print("\n")
    print("=" * 60)
    print("USER QUESTION")
    print("=" * 60)

    print(query)


    # =====================================================
    # RETRIEVAL
    # =====================================================

    results = retrieve(

        query=query,

        documents=documents,

        api_key=api_key,

        top_k=3,

        min_score=0.50
    )


    # =====================================================
    # CHECK RETRIEVAL RESULTS
    # =====================================================

    if not results:

        print("\n")
        print("=" * 60)
        print("NO RELEVANT CONTEXT")
        print("=" * 60)

        print(
            "No sufficiently relevant chunks "
            "were found in the transcript."
        )

        return


    # =====================================================
    # DISPLAY RETRIEVED CHUNKS
    # =====================================================

    print("\n")
    print("=" * 60)
    print("RETRIEVED CHUNKS")
    print("=" * 60)


    for rank, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        print("\n")
        print("-" * 60)

        print(
            f"Rank       : {rank}"
        )

        print(
            f"Chunk ID    : {result['chunk_id']}"
        )

        print(
            f"Similarity : {result['score']:.4f}"
        )

        print(
            f"Timestamp  : "
            f"{metadata['start']} --> "
            f"{metadata['end']}"
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
            f"{len(result['text'])}"
        )


    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    context = build_context(
        results
    )


    # =====================================================
    # BUILD RAG PROMPT
    # =====================================================

    prompt = build_prompt(

        query=query,

        context=context
    )


    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    print("\n")
    print(
        "Generating answer..."
    )


    answer = generate_answer(

        prompt=prompt,

        api_key=api_key
    )


    # =====================================================
    # FINAL ANSWER
    # =====================================================

    print("\n")
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)

    # =====================================================
    # DISPLAY SOURCES
    # =====================================================

    display_sources(results)

if __name__ == "__main__":
    main()    


