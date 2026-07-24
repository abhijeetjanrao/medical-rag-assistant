# Medical RAG Assistant

[![CI](https://github.com/abhijeetjanrao/medical-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/abhijeetjanrao/medical-rag-assistant/actions/workflows/ci.yml)

A retrieval-augmented chatbot that answers clinical questions grounded in WHO/CDC
guidelines and medical reference texts, with source citations and safety
guardrails for a medical-adjacent domain.

What I built / My role
----------------------

A hybrid retrieval-augmented generator (RAG) that combines dense vector
search with BM25 lexical search, applying rule-based guardrails for
medical-adjacent queries. I implemented the backend API, ingestion pipeline
for PDFs → embeddings → ChromaDB, and a Streamlit frontend UI for interactive
chat and source-citation display.

Tech stack
----------

- **Backend:** FastAPI, ChromaDB, LangChain
- **Frontend:** Streamlit
- **Dev / infra:** Docker Compose, PostgreSQL (prod), SQLite (dev)
- **Testing / Eval:** pytest, RAGAS

Demo
----

Add a short demo GIF at `assets/demo.gif` (10–20s) and it will display here:

![Demo placeholder](assets/demo.gif)

## Architecture

```
User -> React/Streamlit Frontend -> FastAPI Backend
                                        |
                    +-------------------+-------------------+
                    |                                       |
              Guardrails layer                      Hybrid RAG Pipeline
           (emergency/diagnosis                    (vector + BM25 search,
              detection)                             reciprocal rank fusion)
                    |                                       |
                    +-------------------+-------------------+
                                        |
                              Gemini (grounded generation)
                                        |
                          SQLite/Postgres (conversation + citation log)
```

## Why these design decisions

**Hybrid retrieval (vector + BM25), not vector search alone.**
Pure semantic search misses exact terms clinicians and patients actually
search for — drug names, dosages, ICD codes. BM25 catches lexical matches;
dense retrieval catches semantic ones. Results are combined with Reciprocal
Rank Fusion, which avoids the problem of normalizing two very different
score scales (cosine similarity vs. BM25 scores).

**Rule-based guardrails, not an extra LLM call.**
An LLM-based safety check adds latency, cost, and a second point of failure.
A small set of explainable regex-based checks catches the clearest cases
(emergency symptoms, personal diagnosis requests) fast and cheaply, and every
flag is auditable — you can point to exactly which rule fired.

**Every answer is traceable to a source document and page number.**
For medical content, an ungrounded answer is a liability. Every retrieved
chunk that contributed to an answer is logged with its source document, page
number, and relevance score, and surfaced to the user in the UI.

**SQLite for dev, Postgres for prod.**
Zero-setup local development, with a straightforward swap to Postgres via
`DB_URL` for concurrent writes and real backups in production.

## Evaluation

Measured on a held-out set of clinical Q&A pairs using
[RAGAS](https://github.com/explodinggym/ragas) plus a custom retrieval-precision
check:

| Metric | Result |
|---|---|
| Retrieval precision@k | *run `eval/evaluate.py` to populate* |
| Faithfulness (RAGAS) | *run `eval/evaluate.py` to populate* |
| Answer relevancy (RAGAS) | *run `eval/evaluate.py` to populate* |
| Latency p50 / p95 | *run `eval/evaluate.py` to populate* |

Run `python -m eval.evaluate` against a live backend to populate this table
with real numbers — those are the numbers to put on a resume, not placeholders.

## Project structure

```
medical-rag-bot/
├── backend/
│   ├── main.py                 # FastAPI app, /chat endpoint
│   ├── rag/
│   │   ├── ingest.py            # PDF -> chunks -> embeddings -> ChromaDB
│   │   ├── retrieve.py          # hybrid vector + BM25 retrieval
│   │   └── guardrails.py        # safety layer
│   ├── db/models.py             # SQLAlchemy models
│   ├── eval/
│   │   ├── evaluate.py          # RAGAS + retrieval precision harness
│   │   └── eval_set.json        # labeled eval questions
│   ├── tests/                   # pytest unit tests
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py                   # Streamlit chat UI
│   └── Dockerfile
├── data/medical_pdfs/           # source documents (not committed)
├── docker-compose.yml
└── .github/workflows/ci.yml     # lint + test + build on push
```

## Running locally

You can run the project with Docker Compose (recommended) or locally using virtual environments. Follow the commands below for a reproducible setup.

Option A — Docker Compose (recommended)

Prereq: Docker Desktop (or Docker Engine + docker-compose).

```bash
# from repo root
docker-compose up --build
# Frontend: http://localhost:8501
# API: http://localhost:8000
```

Option B — Local (macOS / Linux)

```bash
# 1. Add source PDFs (replace with your path)
cp /path/to/your/pdfs/*.pdf data/medical_pdfs/

# 2. Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit `backend/.env` and set GEMINI_API_KEY

# 3. Index the documents (run once, and whenever docs change)
python -m rag.ingest

# 4. Run the API
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 5. Frontend (separate terminal)
cd ../frontend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Option B — Local (Windows PowerShell)

```powershell
# 1. Copy PDFs into the project (adjust source path)
Copy-Item -Path C:\path\to\your\pdfs\*.pdf -Destination .\data\medical_pdfs\

# 2. Backend setup
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit backend\.env and set GEMINI_API_KEY (open in editor)

# 3. Index the documents
python -m rag.ingest

# 4. Run the API
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 5. Frontend (separate terminal)
cd ..\frontend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Notes
- The repo uses SQLite by default (`DB_URL=sqlite:///./medical_rag.db` in `backend/.env`). Docker Compose uses Postgres via `DB_URL` override.
- Do NOT commit secrets. Keep `backend/.env` local and filled from `backend/.env.example`.
- If you host the frontend, set `API_URL` in the frontend environment to your deployed backend URL.

## Running the evaluation suite

```bash
cd backend
pytest tests/ -v            # unit tests
python -m eval.evaluate     # RAGAS + retrieval metrics against a live backend
```

## Tradeoffs and future work

- **BM25 index is in-memory**, rebuilt from the full corpus at backend
  startup. Fine for a few hundred thousand chunks; would move to
  Elasticsearch/OpenSearch beyond that.
- **Guardrails are rule-based**, which is fast and explainable but will miss
  paraphrased emergency language. A fine-tuned classifier would generalize
  better at the cost of latency and complexity.
- **No authentication/rate limiting yet** — required before any real
  deployment handling patient-adjacent queries.
- **Embedding model is a general biomedical model**
  (`pritamdeka/S-PubMedBert-MS-MARCO`); fine-tuning on the specific document
  corpus would likely improve retrieval precision further.
