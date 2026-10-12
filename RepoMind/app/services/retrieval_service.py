
from app.services.embedding_service import generate_embedding
from app.services.vector_store import get_pinecone_index


async def retrieve_repository_chunks(
    repository_id: str,
    question: str,
    top_k: int = 5,
) -> list[dict]:
    if not repository_id.strip():
        raise ValueError("repository_id cannot be empty")

    if not question.strip():
        raise ValueError("question cannot be empty")

    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")

    # 1. Convert the question into an embedding.
    question_embedding = await generate_embedding(question)

    # 2. Search only within this repository's namespace.
    index = get_pinecone_index()
    response = index.query(
        vector=question_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=repository_id,
    )

    # 3. Convert Pinecone matches into ordinary Python dictionaries.
    results = []

    for match in response.matches:
        metadata = match.metadata or {}

        results.append(
            {
                "source": metadata.get("source"),
                "chunk_index": metadata.get("chunk_index"),
                "content": metadata.get("content"),
                "score": match.score,
            }
        )

    return results
