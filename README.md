# Enterprise Text-to-SQL API (NST Challenge)

FastAPI microservice that:

- Retrieves relevant Beaver schema tables via semantic search
- Generates SQL via an LLM using retrieved schema context
- Validates and executes SQL against a local SQLite schema
- Benchmarks retrieval + generation performance (optional in the challenge; implemented here)

## Requirements

- Python 3.11+
- Beaver dataset files, either:
  - Local folders containing the downloaded parquet files (recommended), or
  - Hugging Face access to `beaverbench/beaver-table` and `beaverbench/beaver-query`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set environment variables:

- If using Hugging Face download:
  - `HF_TOKEN` (required unless you already ran `huggingface-cli login`)
- If using local files (your current setup):
  - `LOCAL_TABLE_DIR=../table`
  - `LOCAL_QUERY_DIR=../query`
- One LLM key (free/paid):
  - `GROQ_API_KEY` (recommended default provider)
  - or `OPENROUTER_API_KEY`
  - or `TOGETHER_API_KEY`
  - or `OPENAI_API_KEY`

Optional:

- `LLM_PROVIDER` = `groq` | `openrouter` | `together` | `openai` | `ollama`
- `LLM_MODEL` (default: `llama-3.1-70b-versatile`)

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

- Swagger UI: http://127.0.0.1:8000/docs

## API

### POST /retrieve

```bash
curl -s -X POST http://127.0.0.1:8000/retrieve \\
  -H 'Content-Type: application/json' \\
  -d '{\"question\":\"Which departments have more than 100 students?\"}' | jq
```

### POST /generate-sql

```bash
curl -s -X POST http://127.0.0.1:8000/generate-sql \\
  -H 'Content-Type: application/json' \\
  -d '{\"question\":\"Which departments have more than 100 students?\",\"use_retrieved_context\":true}' | jq
```

### POST /benchmark

```bash
curl -s -X POST http://127.0.0.1:8000/benchmark | jq
```

## Notes

- The service builds a local SQLite database schema from Beaver table metadata and executes queries for validation. If Beaver data rows are not available, result sets can be empty; execution still catches missing-table/column errors.
- Logs are structured JSON and include the full prompt + LLM response for debugging.
