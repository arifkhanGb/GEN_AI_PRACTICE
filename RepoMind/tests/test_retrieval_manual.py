
import asyncio

from app.services.retrieval_service import retrieve_repository_chunks


async def main():
    # Replace this with the repository you successfully indexed.
    repository_id = "arifkhanGb/FASTAPI-FOUNDATION"
   
    question = "Where is the application entry point configured?"

    results = await retrieve_repository_chunks(
        repository_id=repository_id,
        question=question,
        top_k=5,
    )

    if not results:
        print("No matching chunks found.")
        return

    print(f"Found {len(results)} matching chunks.\n")

    for number, result in enumerate(results, start=1):
        print(f"Result {number}")
        print(f"File: {result['source']}")
        print(f"Chunk: {result['chunk_index']}")
        print(f"Score: {result['score']}")
        print(f"Content:\n{result['content'][:500]}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
