"""
app/generation/rag_answer.py  —  STEP 8 CORE

============================================================
LESSON: Put the full RAG loop together
============================================================

Document Parsing → Chunking → Embeddings → Retrieval → LLM

Step 8 is where users FEEL the product:
  question in → grounded answer out

But remember the course mantra:
  If Steps 1–7 are weak, Step 8 cannot save you.

What we implement:
  1. retrieve(question)                 # Step 7
  2. build grounded prompt + call LLM   # Step 8
  3. return answer + citations + hits   # observability for debugging
"""

from __future__ import annotations

from app.generation.llm_client import LlmConfigError, generate_answer_text
from app.models.answer import RagAnswer
from app.retrieval.retriever import retrieve


def answer_question(
    question: str,
    *,
    top_k: int = 3,
    document_ids: list[str] | None = None,
    min_score: float = 0.15,
) -> RagAnswer:
    """
    End-to-end RAG ask: retrieve → real LLM (OpenAI) → citations.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    retrieval = retrieve(
        question,
        top_k=top_k,
        document_ids=document_ids,
        min_score=min_score,
    )

    try:
        answer_text, provider, model = generate_answer_text(question, retrieval)
    except LlmConfigError:
        raise
    except Exception as exc:
        # Surface API errors clearly (quota, invalid key, network)
        raise RuntimeError(f"LLM API call failed: {exc}") from exc

    refusal = "could not find that in the provided documents" in answer_text.lower()

    citations = []
    for i, hit in enumerate(retrieval.hits, start=1):
        citations.append(
            {
                "source": i,
                "chunk_id": hit.chunk_id,
                "file": hit.source_filename,
                "pages": hit.page_numbers,
                "score": hit.score,
                "chunk_type": hit.chunk_type,
            }
        )

    return RagAnswer(
        question=question,
        answer=answer_text,
        citations=citations,
        retrieved_hits=retrieval.hits,
        context_used=retrieval.context_for_llm,
        llm_provider=provider,
        llm_model=model,
        grounded=True,
        refusal=refusal,
    )
