"""
app/generation/llm_client.py

Real OpenAI-compatible Chat Completions for Step 8.
"""

from __future__ import annotations

from app.core.config import settings
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from app.models.retrieval import RetrievalResult


class LlmConfigError(RuntimeError):
    """Raised when a real LLM is required but not configured."""


def _openai_available() -> bool:
    key = (settings.openai_api_key or "").strip()
    return bool(key) and key.lower() not in {"", "your-key-here", "changeme"}


def generate_with_openai(question: str, context: str) -> tuple[str, str, str]:
    """
    Real API call to OpenAI (or any OpenAI-compatible base_url).

    WHY temperature=0?
    RAG answers should stick to retrieved facts, not be creative.
    """
    from openai import OpenAI

    if not _openai_available():
        raise LlmConfigError(
            "OPENAI_API_KEY missing. Put it in GEN_AI_PRACTICE/.env"
        )

    client = OpenAI(
        api_key=settings.openai_api_key.strip(),
        base_url=(settings.openai_base_url or "https://api.openai.com/v1").strip(),
    )
    model = settings.llm_model
    user_prompt = build_user_prompt(question, context)

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("LLM returned an empty response")
    return text, "openai", model


def generate_extractive_fallback(question: str, retrieval: RetrievalResult) -> tuple[str, str, str]:
    """Offline grounded fallback — only used when REQUIRE_REAL_LLM=false."""
    if not retrieval.hits:
        return (
            "I could not find that in the provided documents.\n\nCitations: none",
            "extractive_fallback",
            "none",
        )

    lines = ["Based only on the retrieved document context:", ""]
    top = retrieval.hits[0]
    lines.append(top.text.strip())
    lines.append("")

    if len(retrieval.hits) > 1 and retrieval.hits[1].score >= 0.35:
        second = retrieval.hits[1]
        if second.text.strip()[:80] not in top.text:
            lines.append("Additional related context:")
            lines.append(second.text.strip())
            lines.append("")

    lines.append("Citations:")
    for i, hit in enumerate(retrieval.hits, start=1):
        pages = ",".join(str(p) for p in hit.page_numbers) or "?"
        lines.append(
            f"- Source {i}: {hit.source_filename} (pages={pages}, score={hit.score:.3f})"
        )
    return "\n".join(lines), "extractive_fallback", "none"


def generate_answer_text(question: str, retrieval: RetrievalResult) -> tuple[str, str, str]:
    """
    Prefer real OpenAI API.
    If REQUIRE_REAL_LLM=true (default now), never silently fall back.
    """
    context = retrieval.context_for_llm

    if _openai_available():
        return generate_with_openai(question, context)

    if settings.require_real_llm:
        raise LlmConfigError(
            "Real LLM required but OPENAI_API_KEY is empty. "
            "Add it to GEN_AI_PRACTICE/.env (see .env.example)."
        )

    return generate_extractive_fallback(question, retrieval)
