
from functools import lru_cache

from pinecone import Pinecone

from app.config import get_settings


@lru_cache
def get_pinecone_index():
    settings = get_settings()

    if not settings.pinecone_api_key:
        raise ValueError("Pinecone API key is not configured")

    client = Pinecone(api_key=settings.pinecone_api_key)

    return client.Index(settings.pinecone_index_name)


def upsert_chunks(
    repository_id: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have exactly one embedding")

    if not chunks:
        return 0

    index = get_pinecone_index()
    vectors = []

    for chunk, embedding in zip(chunks, embeddings):
        vector_id = (
            f"{repository_id}:{chunk['source']}:{chunk['chunk_index']}"
        )

        vectors.append(
            {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "repository_id": repository_id,
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                },
            }
        )

    # Pinecone accepts batches; send up to 100 vectors per request.
    for start in range(0, len(vectors), 100):
        index.upsert(
            vectors=vectors[start : start + 100],
            namespace=repository_id,
        )

    return len(vectors)
