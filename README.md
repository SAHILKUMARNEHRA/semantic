# Enterprise Text-to-SQL API (NST Challenge)

FastAPI microservice that:

- Retrieves relevant Beaver schema tables via semantic search
- Generates SQL via an LLM using retrieved schema context
- Validates and executes SQL against a local SQLite schema
- Benchmarks retrieval + generation performance (optional in the challenge; implemented here)

## Architecture

Pipeline:

1. Retrieval layer selects relevant tables from Beaver schema for the user question.
2. Prompt builder constructs the final LLM prompt using retrieved schemas and few-shot examples.
3. LLM layer generates a single SQLite-compatible SQL query.
4. Validation layer checks SQL syntax and executes it against a local SQLite schema for sanity checks.
5. Metrics layer (benchmark) evaluates retrieval and SQL quality over a sample of Beaver benchmark questions.

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
- `GROQ_MODEL` (optional override when `LLM_PROVIDER=groq`)

Note: If you use Groq free tier, `/benchmark` can take longer due to API rate limits.

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

## Submission Checklist (NST)

- Repository includes all source code and this README.
- Screenshots demonstrate the service running locally and responses for:
  - `/docs` (Swagger UI loaded)
  - `POST /retrieve` (request + 200 response)
  - `POST /generate-sql` (request + 200 response)
  - `POST /benchmark` (200 response; may take longer due to rate limits)
- Store screenshots under `screenshots/` and include them in the repo before submitting.
