
import httpx


GITHUB_API_BASE_URL = "https://api.github.com"


async def fetch_repository_files(
    owner: str,
    repository: str,
) -> list[dict]:
    """Fetch the file tree for a GitHub repository."""

    url = (
        f"{GITHUB_API_BASE_URL}/repos/"
        f"{owner}/{repository}/git/trees/HEAD"
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            url,
            params={"recursive": "1"},
            headers={
                "Accept": "application/vnd.github+json",
            },
        )

        response.raise_for_status()
        data = response.json()

    return [
        item
        for item in data.get("tree", [])
        if item.get("type") == "blob"
    ]
    
    
async def fetch_file_content(
    owner: str,
    repository: str,
    path: str,
) -> str:
    """Download a text file from a GitHub repository."""

    url = (
        f"{GITHUB_API_BASE_URL}/repos/"
        f"{owner}/{repository}/contents/{path}"
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            url,
            headers={
                "Accept": "application/vnd.github.raw+json",
            },
        )

        response.raise_for_status()
        return response.text

