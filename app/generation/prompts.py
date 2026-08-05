"""
app/generation/prompts.py

============================================================
LESSON: The prompt is LAST — but still critical
============================================================

Your LinkedIn lesson was right:
  "The quality of your RAG system is determined long before the prompt."

But a BAD prompt can STILL ruin good retrieval:
  - model invents facts not in context (hallucination)
  - model ignores citations
  - model over-answers from world knowledge

Production RAG prompts enforce:
  1) Answer ONLY from provided context
  2) If missing → say you don't know
  3) Cite sources (page / file)
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a careful enterprise document assistant.

Rules (non-negotiable):
1. Use ONLY the information inside CONTEXT.
2. If CONTEXT is insufficient, say exactly: "I could not find that in the provided documents."
3. Do NOT use outside knowledge.
4. Be concise and factual.
5. End with a Citations section listing Source numbers you used.
"""


def build_user_prompt(question: str, context: str) -> str:
    """
    Assemble the user message for the LLM.

    WHY separate system vs user?
    System = permanent behavior rules.
    User = this question + this retrieved context.
    """
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer the QUESTION using only CONTEXT. Include Citations."
    )
