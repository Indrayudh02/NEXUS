# NEXSUS: Knowledge Gap Discovery Engine

The existing vanilla HTML/CSS/JS interface is backed by a FastAPI service in live mode. PDFs are parsed with PyMuPDF; SQLite stores documents, page-preserving chunks, embeddings, extracted graph state, gaps, investigations, and evidence links. The old seeded flow remains available only after explicitly switching to Demo Mode in the sidebar.

## Run the complete system

From this directory, in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` and `GEMINI_MODEL=gemini-2.5-flash` in `.env` for extraction and investigation. By default `EMBEDDING_PROVIDER=local` loads the Sentence Transformers model `all-MiniLM-L6-v2` locally (384-dimensional normalized vectors); the model is downloaded/cached by Sentence Transformers on first use. Local embeddings require no embedding API credits. For external vectors, set `EMBEDDING_PROVIDER=external` and configure `EMBEDDING_MODEL`, `LLM_API_KEY`, and optional `LLM_BASE_URL` for the existing OpenAI-compatible embeddings endpoint. The LLM pipeline does not silently substitute fake vectors or Demo Mode.

Start the API in one terminal:

```powershell
uvicorn backend:app --reload --port 8001
```

Start the existing frontend in a second terminal:

```powershell
python -m http.server 8000
```

Open http://localhost:8000. The frontend calls `http://127.0.0.1:8001`; set `window.NEXUS_API_BASE` before `app.js` if the API is hosted elsewhere. SQLite and uploaded PDFs are created under `data/`.

## APIs

- `GET /health`, `GET /state`
- `POST /documents/upload`, `GET /documents`, `GET /documents/{id}`, `POST /documents/{id}/process`
- `GET /knowledge-graph`, `GET /knowledge-gaps`, `GET /evidence`, `GET /retrieval/search?query=...`, `POST /knowledge-graph/update`
- `POST /investigations/start`, `POST /investigations/{id}/run`, `GET /investigations`, `GET /investigations/{id}`, `GET /investigations/{id}/evidence`, `GET /investigations/{id}/verification`

## Verification

```powershell
python -m unittest discover -s tests -v
node --check app.js
```

The LLM extraction/investigation pipeline requires network access and valid API credentials. The local embedding model requires network access for its first download unless already cached. Scanned PDFs are rejected when they contain no extractable text; OCR is not configured. Page-numbered text chunks are committed before embedding; if embedding fails, the document becomes `FAILED` and those chunks remain available for `POST /documents/{id}/process` to retry. Vectors are persisted in SQLite and ranked with cosine similarity, avoiding a separate vector database service for this prototype.
