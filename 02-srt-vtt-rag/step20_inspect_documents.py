import json
from pathlib import Path


def main():

    path = Path("output/embedded_chunks.json")

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    documents = data["documents"]

    print()
    print("=" * 70)
    print("DOCUMENT STRUCTURE")
    print("=" * 70)

    print()
    print(f"Total documents: {len(documents)}")

    if not documents:
        print("No documents found.")
        return

    document = documents[0]

    print()
    print("FIRST DOCUMENT KEYS")
    print("-" * 70)

    for key in document.keys():
        print(key)

    print()
    print("=" * 70)
    print("FIRST DOCUMENT")
    print("=" * 70)

    print(
        json.dumps(
            document,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()