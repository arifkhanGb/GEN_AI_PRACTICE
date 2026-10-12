
import asyncio

from app.services.github import fetch_file_content


async def main():
    content = await fetch_file_content(
        owner="fastapi",
        repository="fastapi",
        path="README.md",
    )

    print(content[:1000])


if __name__ == "__main__":
    asyncio.run(main())
