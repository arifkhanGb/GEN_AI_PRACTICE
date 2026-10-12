
from openai import AsyncOpenAI

from app.config import get_settings


settings = get_settings()

client = AsyncOpenAI(
    api_key=settings.openai_api_key,
)


async def generate_embedding(text: str) -> list[float]:
    """Generate an embedding vector for a single text."""

    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    return response.data[0].embedding



async def generate_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """Generate embeddings for multiple texts in one API request."""

    if not texts:
        return []

    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )

    ordered_data = sorted(response.data, key=lambda item: item.index)

    return [item.embedding for item in ordered_data]

