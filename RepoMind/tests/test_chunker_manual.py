
from app.services.chunker import chunk_text


text = "A" * 2500

chunks = chunk_text(
    text,
    chunk_size=1000,
    chunk_overlap=200,
)

print("Original characters:", len(text))
print("Number of chunks:", len(chunks))
print("Chunk lengths:", [len(chunk) for chunk in chunks])

assert len(chunks) == 3
assert [len(chunk) for chunk in chunks] == [1000, 1000, 900]
assert chunks[0][-200:] == chunks[1][:200]

print("Chunking test passed!")
