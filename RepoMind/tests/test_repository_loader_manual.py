
import asyncio

from app.services.file_filter import filter_repository_files
from app.services.github import fetch_repository_files
from app.services.repository_loader import load_repository_files


async def main():
    files = await fetch_repository_files("fastapi", "fastapi")
    filtered_files = filter_repository_files(files)

    loaded_files = await load_repository_files(
        owner="fastapi",
        repository="fastapi",
        files=filtered_files,
        limit=10,
    )

    print(f"Filtered files: {len(filtered_files)}")
    print(f"Downloaded files: {len(loaded_files)}")

    for file in loaded_files:
        print(f"\n--- {file['path']} ---")
        print(file["content"][:200])


if __name__ == "__main__":
    asyncio.run(main())
