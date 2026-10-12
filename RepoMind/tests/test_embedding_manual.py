

import asyncio

from app.services.embedding_service import generate_embedding


async def main():
    text = "FastAPI is a Python web framework."

    embedding = await generate_embedding(text)

    print("Embedding dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])

    assert len(embedding) == 1536
    print("Embedding test passed!")


if __name__ == "__main__":
    asyncio.run(main())
