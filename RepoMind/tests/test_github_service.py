
import asyncio

from app.services.github import fetch_repository_files


async def main():
    files = await fetch_repository_files("fastapi", "fastapi")

    print(f"Total files fetched: {len(files)}")

    for file in files[:10]:
        print(file["path"])


if __name__ == "__main__":
    asyncio.run(main())
