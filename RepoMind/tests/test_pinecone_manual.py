
import asyncio
from uuid import uuid4

from app.services.embedding_service import generate_embedding
from app.services.vector_store import get_pinecone_index


async def main():
    text = "FastAPI is used to build Python web APIs."

    # 1. Convert text into a vector using OpenAI
    embedding = await generate_embedding(text)

    # 2. Connect to Pinecone
    index = get_pinecone_index()

    # 3. Store the vector and its metadata
    vector_id = f"repomind-test-{uuid4().hex[:8]}"

    index.upsert(
        vectors=[
            {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": text,
                    "source": "manual-test",
                },
            }
        ],
        namespace="repomind-test",
    )

    print(f"Vector uploaded: {vector_id}")

    # 4. Search using the same embedding
    results = index.query(
        vector=embedding,
        top_k=1,
        include_metadata=True,
        namespace="repomind-test",
    )

    print("Search results:")
    for match in results.matches:
        print("Score:", match.score)
        print("Metadata:", match.metadata)


if __name__ == "__main__":
    asyncio.run(main())
