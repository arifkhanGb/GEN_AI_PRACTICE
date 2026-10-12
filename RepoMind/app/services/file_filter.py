
from pathlib import PurePosixPath


IGNORED_DIRECTORIES = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    "out",
    "coverage",
    ".next",
    "vendor",
    "target",
    "__pycache__",
    ".gradle",
}

ALLOWED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".md",
    ".mdx",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".html",
    ".css",
    ".sql",
    ".sh",
}

IMPORTANT_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "Procfile",
}


def filter_repository_files(
    files: list[dict],
) -> list[dict]:
    """Keep relevant files and exclude ignored directories."""

    filtered_files = []

    for file in files:
        path = PurePosixPath(file["path"])

        if any(
            part in IGNORED_DIRECTORIES
            for part in path.parts
        ):
            continue

        if (
            path.suffix.lower() in ALLOWED_EXTENSIONS
            or path.name in IMPORTANT_FILENAMES
        ):
            filtered_files.append(file)

    return filtered_files
