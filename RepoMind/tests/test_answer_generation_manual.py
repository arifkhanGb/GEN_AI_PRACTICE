
import asyncio

from app.services.answer_generation_service import (
    generate_repository_answer,
)


async def main():
    result = await generate_repository_answer(
        repository_id = "arifkhanGb/FASTAPI-FOUNDATION",
        question="Where is the application entry point configured?",
        top_k=5,
    )

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nSOURCES:\n")
    for source in result["sources"]:
        print(source)


if __name__ == "__main__":
    asyncio.run(main())
