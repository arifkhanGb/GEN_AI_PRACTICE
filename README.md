# GEN_AI_PRACTICE

Personal GenAI practice monorepo.

Each folder is a **separate project**. Add new experiments as new folders (do not dump files at the repo root).

## Projects

| Folder | What it is |
|--------|------------|
| [`01-rag-pdf-pipeline`](./01-rag-pdf-pipeline) | Production-style RAG: PDF parse → chunk → embed → **Qdrant** → OpenAI answer |

## How to add a new project later

1. Create a new folder, e.g. `02-agents-basics/`
2. Put that project's code, `requirements.txt`, and `.env.example` inside it
3. Keep secrets in that project's `.env` (never commit `.env`)
4. Commit and push

## Quick start (RAG project)

```powershell
cd 01-rag-pdf-pipeline
python -m pip install -r requirements.txt
copy .env.example .env
# put your OPENAI_API_KEY in .env
python scripts/run_parse_local.py data/uploads/sample_step5_chunking.pdf --demo-chunks
python scripts/run_ask.py "What pressure must the wellhead hold during testing?"
```
