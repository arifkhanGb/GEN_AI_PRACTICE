
import asyncio

from app.services.github import fetch_file_content


async def load_repository_files(
    owner: str,
    repository: str,
    files: list[dict],
    limit: int = 10,
) -> list[dict[str, str]]:
    """Download the contents of a limited number of files."""

    selected_files = files[:limit]
    semaphore = asyncio.Semaphore(3)

    async def load_one(file: dict) -> dict[str, str] | None:
        async with semaphore:
            try:
                content = await fetch_file_content(
                    owner=owner,
                    repository=repository,
                    path=file["path"],
                )

                return {
                    "path": file["path"],
                    "content": content,
                }
            except Exception as exc:
                print(f"Failed to load {file['path']}: {exc}")
                return None

    results = await asyncio.gather(
        *(load_one(file) for file in selected_files)
    )

    return [result for result in results if result is not None]

#  Understand the important parts
# - files[:limit] selects the first 10 files for this test.
# - asyncio.Semaphore(3) allows at most three downloads concurrently.
# - asyncio.gather() runs the selected downloads concurrently.
# - Failed downloads are skipped, while successful results contain the path and content.