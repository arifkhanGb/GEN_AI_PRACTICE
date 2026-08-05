# Project README — 01-rag-pdf-pipeline

Production-style **RAG** learning project:

`PDF → parse/tables/OCR/clean → chunk → embed → Qdrant → retrieve → OpenAI`

## Setup

```powershell
cd 01-rag-pdf-pipeline
python -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`.

## Run

```powershell
# Index (Phase 1)
python scripts/run_parse_local.py data/uploads/sample_step5_chunking.pdf --demo-chunks

# Ask (Phase 2)
python scripts/run_ask.py "What pressure must the wellhead hold during testing?"

# API
python -m uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs
