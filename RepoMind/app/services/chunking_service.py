
from app.services.chunker import chunk_text


def chunk_repository_files(
    files: list[dict[str, str]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, str | int]]:
    """Chunk loaded files while preserving source metadata."""

    repository_chunks = []

    for file in files:
        chunks = chunk_text(
            text=file["content"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for index, chunk in enumerate(chunks):
            repository_chunks.append(
                {
                    "source": file["path"],
                    "chunk_index": index,
                    "content": chunk,
                }
            )

    return repository_chunks

# Each chunk now contains:
# - source: the original repository file path.
# - chunk_index: the chunk's position within that file, starting at zero.
# - content: the actual text used later for embeddings and retrieval.
