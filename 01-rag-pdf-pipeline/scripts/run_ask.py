"""
scripts/run_ask.py

Full RAG ask (Step 7 + 8):

    python scripts/run_ask.py "What pressure must the wellhead hold during testing?"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.generation.rag_answer import answer_question


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/run_ask.py "<question>"')
        sys.exit(1)

    question = sys.argv[1]
    result = answer_question(question, top_k=3)

    print("=" * 60)
    print("STEP 8 - RAG Answer")
    print("=" * 60)
    print(f"provider : {result.llm_provider}")
    print(f"model    : {result.llm_model}")
    print(f"question : {result.question}")
    print()
    print("--- ANSWER ---")
    print(result.answer)
    print()
    print("--- CITATIONS ---")
    for c in result.citations:
        print(f"  Source {c['source']}: {c['file']} pages={c['pages']} score={c['score']}")
    print()
    print("Pipeline complete: Parse -> Chunk -> Embed -> Retrieve -> Answer")


if __name__ == "__main__":
    main()
