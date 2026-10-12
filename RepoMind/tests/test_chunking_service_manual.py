
from app.services.chunking_service import chunk_repository_files

files = [
    {
        "path": "README.md",
        "content": "A" * 2500,
    },
    {
        "path": "src/main.py",
        "content": "print('Hello RepoMind')",
    },
]

chunks = chunk_repository_files(
    files,
    chunk_size=1000,
    chunk_overlap=200,
)

print("Total chunks:", len(chunks))

for chunk in chunks:
    print(
        chunk["source"],
        chunk["chunk_index"],
        len(chunk["content"]),
    )

assert len(chunks) == 4
assert chunks[0]["source"] == "README.md"
assert chunks[0]["chunk_index"] == 0
assert chunks[-1]["source"] == "src/main.py"

print("Chunking service test passed!")

