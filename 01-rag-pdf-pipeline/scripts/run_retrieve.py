"""
scripts/run_retrieve.py

Demo Step 7 without the API:

    python scripts/run_retrieve.py "What pressure is required for the wellhead test?"
    python scripts/run_retrieve.py "muster point for drills" --top-k 2
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval.retriever import retrieve


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/run_retrieve.py "<question>" [--top-k 3]')
        sys.exit(1)

    query = sys.argv[1]
    top_k = 3
    if "--top-k" in sys.argv:
        i = sys.argv.index("--top-k")
        top_k = int(sys.argv[i + 1])

    result = retrieve(query, top_k=top_k)

    print("=" * 60)
    print("STEP 7 - Retrieval via Qdrant")
    print("=" * 60)
    print(f"query : {result.query}")
    print(f"model : {result.embedding_model}")
    print(f"hits  : {result.hit_count}")
    print()

    for i, hit in enumerate(result.hits, start=1):
        print(f"#{i} score={hit.score:.4f} pages={hit.page_numbers} type={hit.chunk_type}")
        print(f"    file={hit.source_filename}")
        print(f"    text={hit.text[:220]!r}")
        print()

    print("--- context_for_llm (feeds Step 8) ---")
    print(result.context_for_llm[:800])
    print()
    print("When ready, say CONTINUE for Step 8 (LLM answers).")


if __name__ == "__main__":
    main()
