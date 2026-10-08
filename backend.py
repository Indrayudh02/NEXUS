import json
import math
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pymupdf as fitz
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types as genai_types
from openai import APIError as OpenAIAPIError, OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

ROOT = Path(__file__).resolve().parent
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LOCAL_EMBEDDING_DIMENSIONS = 384
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-3.7-flash"
DATA_DIR = Path(os.getenv("NEXUS_DATA_DIR", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "nexus.sqlite3"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NEXSUS Knowledge Engine", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def database():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    with database() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL,
                status TEXT NOT NULL, page_count INTEGER NOT NULL DEFAULT 0,
                error TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL, text TEXT NOT NULL, embedding TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS concept_evidence (
                concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
                chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                PRIMARY KEY(concept_id, chunk_id)
            );
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES concepts(id),
                target_id TEXT NOT NULL REFERENCES concepts(id), relationship_type TEXT NOT NULL,
                confidence REAL NOT NULL, status TEXT NOT NULL, claim TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                UNIQUE(source_id, target_id, relationship_type)
            );
            CREATE TABLE IF NOT EXISTS relationship_evidence (
                relationship_id TEXT NOT NULL REFERENCES relationships(id) ON DELETE CASCADE,
                chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                PRIMARY KEY(relationship_id, chunk_id)
            );
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY, text TEXT NOT NULL, chunk_id TEXT NOT NULL REFERENCES chunks(id),
                status TEXT NOT NULL DEFAULT 'UNVERIFIED', created_at TEXT NOT NULL,
                UNIQUE(text, chunk_id)
            );
            CREATE TABLE IF NOT EXISTS gaps (
                id TEXT PRIMARY KEY, signature TEXT NOT NULL UNIQUE, type TEXT NOT NULL,
                description TEXT NOT NULL, related_concepts TEXT NOT NULL, reason TEXT NOT NULL,
                priority TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'OPEN', created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS investigations (
                id TEXT PRIMARY KEY, gap_id TEXT REFERENCES gaps(id), question TEXT NOT NULL DEFAULT '',
                hypothesis TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'RUNNING',
                confidence REAL NOT NULL DEFAULT 0, stage TEXT NOT NULL DEFAULT 'GAP',
                reasoning_summary TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS investigation_evidence (
                investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
                chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                evidence_type TEXT NOT NULL,
                relevance REAL NOT NULL,
                PRIMARY KEY(investigation_id, chunk_id, evidence_type)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON chunks(document_id);

            CREATE INDEX IF NOT EXISTS idx_investigations_created
                ON investigations(created_at DESC);
            """
        )

        # Lightweight schema migration for document-scoped gaps/investigations.
        for table in ("gaps", "investigations"):
            columns = {
                row["name"]
                for row in db.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            }

            if "document_id" not in columns:
                db.execute(
                    f"ALTER TABLE {table} ADD COLUMN document_id TEXT"
                )


initialize_database()


def require_llm() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(
            503,
            "GEMINI_API_KEY is missing. Configure it in .env to process documents or investigate."
        )

    return genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(
            retry_options=genai_types.HttpRetryOptions(
                attempts=3,
                initial_delay=1.0,
                max_delay=4.0,
                exp_base=2.0,
                jitter=0.5,
                http_status_codes=[429, 503],
            ),
        ),
    )


def require_fallback_llm() -> genai.Client:
    api_key = os.getenv("GEMINI_FALLBACK_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(
            503,
            "GEMINI_FALLBACK_API_KEY is missing. Configure it in .env."
        )

    return genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(
            retry_options=genai_types.HttpRetryOptions(
                attempts=3,
                initial_delay=1.0,
                max_delay=4.0,
                exp_base=2.0,
                jitter=0.5,
                http_status_codes=[429, 503],
            ),
        ),
    )


def model_json(
    client: genai.Client,
    system: str,
    user: str,
    response_schema: type[BaseModel] | None = None,
) -> dict:

    primary_model = (
        os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        or DEFAULT_GEMINI_MODEL
    )

    fallback_model = (
        os.getenv(
            "GEMINI_FALLBACK_MODEL",
            DEFAULT_GEMINI_FALLBACK_MODEL,
        ).strip()
        or DEFAULT_GEMINI_FALLBACK_MODEL
    )

    def make_request(
        active_client: genai.Client,
        active_model: str,
    ) -> dict:
        response = active_client.models.generate_content(
            model=active_model,
            contents=user,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.1,
            ),
        )

        content = response.text or ""

        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                502,
                "Gemini returned malformed JSON. No graph data was saved; retry document processing.",
            ) from exc

        if not isinstance(result, dict):
            raise HTTPException(
                502,
                "Gemini returned JSON that was not an object. No graph data was saved; retry document processing.",
            )

        return result

    try:
        # Primary Gemini key
        return make_request(client, primary_model)

    except HTTPException:
        raise

    except Exception as exc:
        message = str(exc)
        status = str(
            getattr(exc, "code", "")
            or getattr(exc, "status_code", "")
        )

        error_text = f"{status} {message}".lower()

        # Only fail over for transient service/quota errors.
        fallback_markers = (
            "429",
            "resource_exhausted",
            "quota",
            "rate limit",
            "rate_limit",
            "503",
            "unavailable",
            "service unavailable",
        )

        if any(marker in error_text for marker in fallback_markers):
            # Try the second Gemini API key.
            try:
                fallback_client = require_fallback_llm()
                return make_request(fallback_client, fallback_model)

            except HTTPException:
                raise

            except Exception as fallback_exc:
                fallback_message = str(fallback_exc)

                fallback_key = os.getenv(
                    "GEMINI_FALLBACK_API_KEY",
                    "",
                ).strip()

                if fallback_key:
                    fallback_message = fallback_message.replace(
                        fallback_key,
                        "[redacted]",
                    )

                raise HTTPException(
                    502,
                    "Primary Gemini failed and the fallback Gemini key also failed. "
                    f"Fallback details: {fallback_message}",
                ) from fallback_exc

        # Redact the primary key before returning the error.
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

        if api_key:
            message = message.replace(api_key, "[redacted]")

        raise HTTPException(
            502,
            f"Gemini request failed: {message}",
        ) from exc


@lru_cache(maxsize=1)
def local_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise HTTPException(503, "Local embeddings require sentence-transformers. Install dependencies from requirements.txt.") from exc
    try:
        return SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    except Exception as exc:
        raise HTTPException(503, f"Could not load local embedding model {LOCAL_EMBEDDING_MODEL}: {exc}") from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    if provider == "local":
        try:
            vectors = local_embedding_model().encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()
            return [[float(value) for value in vector] for vector in vectors]
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(502, f"Local embedding generation failed: {exc}") from exc
    if provider != "external":
        raise HTTPException(503, f"Unsupported EMBEDDING_PROVIDER '{provider}'. Use 'local' or 'external'.")

    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("EMBEDDING_MODEL", "").strip()
    if not api_key:
        raise HTTPException(503, "LLM_API_KEY is missing for external embeddings.")
    if not model:
        raise HTTPException(503, "EMBEDDING_MODEL is missing for external embeddings.")
    client = OpenAI(api_key=api_key, base_url=os.getenv("LLM_BASE_URL") or None, timeout=60)
    try:
        response = client.embeddings.create(model=model, input=texts)
        return [[float(value) for value in item.embedding] for item in response.data]
    except OpenAIAPIError as exc:
        raise HTTPException(502, f"External embedding request failed: {exc}") from exc


class ExtractedConcept(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=1000)
    evidence_chunk_ids: list[str] = Field(min_length=1)


class ExtractedRelationship(BaseModel):
    source: str = Field(min_length=1, max_length=160)
    target: str = Field(min_length=1, max_length=160)
    type: str = Field(min_length=1, max_length=120)
    confidence: float = Field(ge=0, le=1)
    claim: str = Field(default="", max_length=1200)
    evidence_chunk_ids: list[str] = Field(min_length=1)


class ExtractedClaim(BaseModel):
    text: str = Field(min_length=1, max_length=1200)
    evidence_chunk_id: str


class ExtractedContradiction(BaseModel):
    source: str = Field(min_length=1, max_length=160)
    target: str = Field(min_length=1, max_length=160)
    reason: str = Field(min_length=1, max_length=1200)
    evidence_chunk_ids: list[str]


class Extraction(BaseModel):
    concepts: list[ExtractedConcept]
    relationships: list[ExtractedRelationship]
    claims: list[ExtractedClaim]
    contradictions: list[ExtractedContradiction] = Field(default_factory=list)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_chunks(text: str, limit: int = 1800) -> list[str]:
    paragraphs = [clean_text(p) for p in re.split(r"\n\s*\n", text) if clean_text(p)]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > limit:
            if current:
                chunks.append(current)
                current = ""
            split_at = paragraph.rfind(" ", 0, limit)
            split_at = split_at if split_at > limit // 2 else limit
            chunks.append(paragraph[:split_at].strip())
            paragraph = paragraph[split_at:].strip()
        candidate = f"{current} {paragraph}".strip()
        if len(candidate) > limit and current:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def extract_document(document_id: str) -> None:
    with database() as db:
        document = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if not document:
            return
        db.execute("UPDATE documents SET status='PROCESSING', error=NULL WHERE id=?", (document_id,))
    try:
        pages: list[tuple[int, str]] = []
        with fitz.open(document["path"]) as pdf:
            total_pages = pdf.page_count
            for page_number in range(1, total_pages + 1):
                page = pdf.load_page(page_number - 1)
                text = page.get_text("text")
                if isinstance(text, str) and clean_text(text):
                    pages.append((page_number, text))
        with database() as db:
            db.execute("UPDATE documents SET page_count=? WHERE id=?", (total_pages, document_id))
        if not pages:
            raise ValueError("The PDF contains no extractable text. Scanned PDFs require OCR, which is not configured.")

        with database() as db:
            chunk_rows = [dict(row) for row in db.execute(
                "SELECT id,page_number AS page,text,embedding FROM chunks WHERE document_id=? ORDER BY page_number,rowid",
                (document_id,),
            )]
            if not chunk_rows:
                chunk_rows = [
                    {"id": str(uuid.uuid4()), "page": page_number, "text": part, "embedding": "[]"}
                    for page_number, text in pages
                    for part in split_chunks(text)
                ]
                db.executemany(
                    "INSERT INTO chunks(id,document_id,page_number,text,embedding) VALUES(?,?,?,?,?)",
                    [(row["id"], document_id, row["page"], row["text"], "[]") for row in chunk_rows],
                )

        missing_vectors = [row for row in chunk_rows if row["embedding"] in (None, "", "[]")]
        for offset in range(0, len(missing_vectors), 64):
            group = missing_vectors[offset:offset + 64]
            vectors = embed_texts([row["text"] for row in group])
            if len(vectors) != len(group):
                raise ValueError("Embedding provider returned an unexpected number of vectors.")
            if any(not vector for vector in vectors):
                raise ValueError("Embedding provider returned an empty vector.")
            with database() as db:
                db.executemany("UPDATE chunks SET embedding=? WHERE id=? AND document_id=?",
                               [(json.dumps(vector), row["id"], document_id) for row, vector in zip(group, vectors)])

        with database() as db:
            chunk_rows = [dict(row) for row in db.execute(
                "SELECT id,page_number AS page,text,embedding FROM chunks WHERE document_id=? ORDER BY page_number,rowid",
                (document_id,),
            )]
        if not chunk_rows or any(row["embedding"] in (None, "", "[]") for row in chunk_rows):
            raise ValueError("Document chunks are stored, but one or more embeddings are still missing.")

        client = require_llm()

        extracted: list[tuple[list[dict], Extraction]] = []
        for offset in range(0, len(chunk_rows), 5):
            group = chunk_rows[offset:offset + 5]
            payload = [{"chunk_id": row["id"], "text": row["text"]} for row in group]
            result = model_json(
                client,
                "Extract only directly supported concepts, relationships, and claims from the supplied PDF text. "
                "Return JSON with concepts [{name,description,evidence_chunk_ids}], relationships "
                "[{source,target,type,confidence,claim,evidence_chunk_ids}], and claims "
                "[{text,evidence_chunk_id}], and contradictions "
                "[{source,target,reason,evidence_chunk_ids}]. Record a contradiction only when the supplied "
                "text explicitly reports conflicting findings or limitations about the same relationship. "
                "Use only supplied chunk IDs, quote no material not present, "
                "and return empty arrays when evidence is insufficient.",
                json.dumps(payload),
                response_schema=Extraction,
            )
            parsed = Extraction.model_validate(result)
            allowed_ids = {row["id"] for row in group}
            for concept in parsed.concepts:
                if not set(concept.evidence_chunk_ids) <= allowed_ids:
                    raise ValueError("Concept evidence must reference chunks in the supplied PDF text.")
            for relation in parsed.relationships:
                if not relation.evidence_chunk_ids or not set(relation.evidence_chunk_ids) <= allowed_ids:
                    raise ValueError("Relationship evidence must reference chunks in the supplied PDF text.")
            for claim in parsed.claims:
                if claim.evidence_chunk_id not in allowed_ids:
                    raise ValueError("Claim referenced an unknown evidence chunk.")
            for contradiction in parsed.contradictions:
                if not contradiction.evidence_chunk_ids or not set(contradiction.evidence_chunk_ids) <= allowed_ids:
                    raise ValueError("Contradiction evidence must reference chunks in the supplied PDF text.")
            extracted.append((group, parsed))

        with database() as db:
            for group, parsed in extracted:
                for concept in parsed.concepts:
                    db.execute(
                        "INSERT INTO concepts(id,name,description,created_at) VALUES(?,?,?,?) "
                        "ON CONFLICT(name) DO UPDATE SET description=CASE WHEN excluded.description='' "
                        "THEN concepts.description ELSE excluded.description END",
                        (str(uuid.uuid4()), concept.name.strip(), concept.description.strip(), utc_now()),
                    )
                    saved_concept = db.execute("SELECT id FROM concepts WHERE name=? COLLATE NOCASE", (concept.name.strip(),)).fetchone()
                    for chunk_id in concept.evidence_chunk_ids:
                        db.execute("INSERT OR IGNORE INTO concept_evidence VALUES(?,?)", (saved_concept["id"], chunk_id))
                for relation in parsed.relationships:
                    source = db.execute("SELECT id FROM concepts WHERE name=? COLLATE NOCASE", (relation.source.strip(),)).fetchone()
                    target = db.execute("SELECT id FROM concepts WHERE name=? COLLATE NOCASE", (relation.target.strip(),)).fetchone()
                    if not source or not target:
                        continue
                    db.execute(
                        "INSERT INTO relationships(id,source_id,target_id,relationship_type,confidence,status,claim,created_at,updated_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(source_id,target_id,relationship_type) DO UPDATE SET "
                        "confidence=MAX(relationships.confidence,excluded.confidence), status=excluded.status, "
                        "claim=excluded.claim, updated_at=excluded.updated_at",
                        (str(uuid.uuid4()), source["id"], target["id"], relation.type.strip(), relation.confidence,
                         "SUPPORTED" if relation.confidence >= 0.75 else "WEAK", relation.claim.strip(), utc_now(), utc_now()),
                    )
                    saved = db.execute("SELECT id FROM relationships WHERE source_id=? AND target_id=? AND relationship_type=?",
                                       (source["id"], target["id"], relation.type.strip())).fetchone()
                    for chunk_id in relation.evidence_chunk_ids:
                        db.execute("INSERT OR IGNORE INTO relationship_evidence VALUES(?,?)", (saved["id"], chunk_id))
                for claim in parsed.claims:
                    db.execute("INSERT OR IGNORE INTO claims(id,text,chunk_id,created_at) VALUES(?,?,?,?)",
                               (str(uuid.uuid4()), claim.text.strip(), claim.evidence_chunk_id, utc_now()))
                for contradiction in parsed.contradictions:
                    source = db.execute("SELECT id,name FROM concepts WHERE name=? COLLATE NOCASE", (contradiction.source.strip(),)).fetchone()
                    target = db.execute("SELECT id,name FROM concepts WHERE name=? COLLATE NOCASE", (contradiction.target.strip(),)).fetchone()
                    if not source or not target:
                        continue
                    signature = f"contradiction:{min(source['id'], target['id'])}:{max(source['id'], target['id'])}"
                    gap_id = str(uuid.uuid4())
                    db.execute(
                        "INSERT OR IGNORE INTO gaps(id,signature,type,description,related_concepts,reason,priority,status,created_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?)",
                        (gap_id, signature, "Contradiction", f"Conflicting evidence: {source['name']} ↔ {target['name']}",
                         json.dumps([source["name"], target["name"]]), contradiction.reason.strip(), "High", "OPEN", utc_now()),
                    )
            db.execute("UPDATE documents SET status='READY', page_count=? WHERE id=?", (total_pages, document_id))
        detect_gaps()
    except Exception as exc:
        error = exc.detail if isinstance(exc, HTTPException) else str(exc)
        with database() as db:
            db.execute("UPDATE documents SET status='FAILED', error=? WHERE id=?", (str(error)[:1000], document_id))


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match; check the configured embedding model.")
    denominator = math.sqrt(sum(x * x for x in left) * sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0


def retrieve(
    query: str,
    limit: int = 8,
    document_id: str | None = None,
) -> list[dict]:
    with database() as db:
        if document_id:
            rows = db.execute(
                "SELECT c.*, d.name AS document_name "
                "FROM chunks c JOIN documents d ON d.id=c.document_id "
                "WHERE d.status='READY' "
                "AND c.document_id=? "
                "AND c.embedding NOT IN ('', '[]')",
                (document_id,),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT c.*, d.name AS document_name "
                "FROM chunks c JOIN documents d ON d.id=c.document_id "
                "WHERE d.status='READY' "
                "AND c.embedding NOT IN ('', '[]')"
            ).fetchall()

    if not rows:
        return []

    query_vector = embed_texts([query])[0]

    ranked = []

    for row in rows:
        similarity = cosine(
            query_vector,
            json.loads(row["embedding"])
        )

        ranked.append({
            "id": row["id"],
            "document_id": row["document_id"],
            "document": row["document_name"],
            "page": row["page_number"],
            "text": row["text"],
            "relevance": round(max(0, similarity), 4),
        })

    ranked.sort(
        key=lambda item: item["relevance"],
        reverse=True
    )

    return ranked[:limit]
    

def detect_gaps(document_id: str | None = None) -> list[dict]:
    created: list[dict] = []
    with database() as db:
        existing = {row["signature"] for row in db.execute("SELECT signature FROM gaps WHERE status='OPEN'")}
        concepts = db.execute(
            "SELECT c.id,c.name,COUNT(DISTINCT ch.id) AS mentions FROM concepts c "
            "JOIN chunks ch ON lower(ch.text) LIKE '%' || lower(c.name) || '%' "
            "JOIN documents d ON d.id=ch.document_id AND d.status='READY' GROUP BY c.id HAVING mentions > 0"
        ).fetchall()
        relations = db.execute("SELECT source_id,target_id FROM relationships WHERE status IN ('SUPPORTED','PARTIALLY_SUPPORTED')").fetchall()
        known = {(row["source_id"], row["target_id"]) for row in relations}
        for index, first in enumerate(concepts):
            for second in concepts[index + 1:]:
                if (first["id"], second["id"]) in known or (second["id"], first["id"]) in known:
                    continue
                cooccurs = db.execute(
                    "SELECT 1 FROM chunks ch JOIN documents d ON d.id=ch.document_id AND d.status='READY' "
                    "WHERE lower(ch.text) LIKE '%' || lower(?) || '%' AND lower(ch.text) LIKE '%' || lower(?) || '%' LIMIT 1",
                    (first["name"], second["name"]),
                ).fetchone()
                if not cooccurs:
                    continue
                scope = document_id or "global"
                signature = f"missing:{scope}:{min(first['id'],second['id'])}:{max(first['id'],second['id'])}"
                if signature in existing:
                    continue
                gap = {"id": str(uuid.uuid4()), "type": "Missing Relationship",
                       "description": f"{first['name']} → ??? → {second['name']}",
                       "related_concepts": [first["name"], second["name"]],
                       "reason": "Both concepts co-occur in extracted source text, but no supported relationship is recorded in the knowledge graph.",
                       "priority": "High" if min(first["mentions"], second["mentions"]) > 2 else "Moderate",
                       "status": "OPEN", "signature": signature}
                db.execute("INSERT OR IGNORE INTO gaps(id,signature,type,description,related_concepts,reason,priority,status,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                           (gap["id"], signature, gap["type"], gap["description"], json.dumps(gap["related_concepts"]),
                            gap["reason"], gap["priority"], gap["status"], utc_now()))
                created.append(gap)
                existing.add(signature)

        weak = db.execute(
            "SELECT r.id,s.name AS source,t.name AS target,r.confidence,COUNT(re.chunk_id) AS evidence_count "
            "FROM relationships r JOIN concepts s ON s.id=r.source_id JOIN concepts t ON t.id=r.target_id "
            "LEFT JOIN relationship_evidence re ON re.relationship_id=r.id "
            "WHERE r.status='WEAK' OR r.confidence < 0.6 GROUP BY r.id"
        ).fetchall()
        for relation in weak:
            signature = f"weak:{relation['id']}"
            if signature in existing:
                continue
            gap = {"id": str(uuid.uuid4()), "type": "Weak Evidence",
                   "description": f"{relation['source']} → {relation['target']} has limited support",
                   "related_concepts": [relation["source"], relation["target"]],
                   "reason": f"The extracted relationship has confidence {relation['confidence']:.2f} and {relation['evidence_count']} linked source chunks.",
                   "priority": "Moderate", "status": "OPEN", "signature": signature}
            db.execute("INSERT OR IGNORE INTO gaps(id,signature,type,description,related_concepts,reason,priority,status,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                       (gap["id"], signature, gap["type"], gap["description"], json.dumps(gap["related_concepts"]),
                        gap["reason"], gap["priority"], gap["status"], utc_now()))
            created.append(gap)
            existing.add(signature)
        unverified = db.execute("SELECT id,text FROM claims WHERE status='UNVERIFIED'").fetchall()
        for claim in unverified:
            signature = f"claim:{claim['id']}"
            if signature in existing:
                continue
            gap = {"id": str(uuid.uuid4()), "type": "Unverified Claim", "description": claim["text"],
                   "related_concepts": [], "reason": "This claim was extracted from a source chunk but has not been checked against additional evidence.",
                   "priority": "Moderate", "status": "OPEN", "signature": signature}
            db.execute("INSERT OR IGNORE INTO gaps(id,signature,type,description,related_concepts,reason,priority,status,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                       (gap["id"], signature, gap["type"], gap["description"], "[]", gap["reason"], gap["priority"], "OPEN", utc_now()))
            created.append(gap)
            existing.add(signature)
    return created


def gap_row(row: sqlite3.Row) -> dict:
    return {"id": row["id"], "type": row["type"], "description": row["description"],
            "related_concepts": json.loads(row["related_concepts"]), "reason": row["reason"],
            "priority": row["priority"], "status": row["status"]}


def evidence_for_chunk(chunk_id: str, evidence_type: str, relevance: float) -> dict | None:
    with database() as db:
        row = db.execute("SELECT c.id,c.page_number,c.text,d.name FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=?", (chunk_id,)).fetchone()
    if not row:
        return None
    return {"id": row["id"], "source": row["name"], "document": row["name"], "page": row["page_number"],
            "text": row["text"], "relevance": f"{relevance:.2f}", "relevance_score": relevance,
            "type": evidence_type, "contradiction": evidence_type == "counter"}


def update_stage(investigation_id: str, stage: str) -> None:
    with database() as db:
        db.execute("UPDATE investigations SET stage=? WHERE id=?", (stage, investigation_id))


def run_investigation(investigation_id: str) -> None:
    try:
        client = require_llm()

        with database() as db:
            inv = db.execute(
                "SELECT i.*,g.description,g.related_concepts,g.reason "
                "FROM investigations i JOIN gaps g ON g.id=i.gap_id "
                "WHERE i.id=?",
                (investigation_id,),
            ).fetchone()

        if not inv:
            return

        concepts = json.loads(inv["related_concepts"])

        # ---------------------------------------------------------
        # 1. QUESTION
        # ---------------------------------------------------------
        update_stage(investigation_id, "QUESTION")

        question = ""

        try:
            question_json = model_json(
                client,
                "Generate one focused research question for this corpus gap. "
                "Return JSON {question:string}. Do not assert facts.",
                json.dumps({
                    "gap": inv["description"],
                    "reason": inv["reason"],
                    "concepts": concepts,
                }),
            )

            question = str(
                question_json.get("question", "")
            ).strip()

        except Exception as exc:
            print(f"[NEXSUS] Question generation unavailable: {exc}")

            # Deterministic fallback
            question = (
                f"What evidence in the knowledge base explains or clarifies "
                f"the gap: {inv['description']}?"
            )

        if not question:
            raise ValueError("Could not generate a research question.")

        with database() as db:
            db.execute(
                "UPDATE investigations SET question=? WHERE id=?",
                (question, investigation_id),
            )

        # ---------------------------------------------------------
        # 2. RETRIEVE
        # ---------------------------------------------------------
        update_stage(investigation_id, "RETRIEVE")

        supporting = retrieve(
    question,
    document_id=inv["document_id"],
)

        if not supporting:
            raise ValueError(
                "No document chunks are available for retrieval."
            )

        supporting = [
            item for item in supporting
            if item["relevance"] >= 0.15
        ]

        if not supporting:
            raise ValueError(
                "No sufficiently relevant document evidence was retrieved."
            )

        support_payload = [
            {
                "chunk_id": item["id"],
                "document": item["document"],
                "page": item["page"],
                "text": item["text"],
                "relevance": item["relevance"],
            }
            for item in supporting
        ]

        # ---------------------------------------------------------
        # 3. HYPOTHESIS
        # ---------------------------------------------------------
        update_stage(investigation_id, "HYPOTHESIS")

        hypothesis = ""

        try:
            hypothesis_json = model_json(
                client,
                "Form a concise hypothesis using only supplied source "
                "excerpts. Return JSON {hypothesis:string}. "
                "If the excerpts do not support a hypothesis, state that "
                "explicitly. Do not add outside knowledge or citations.",
                json.dumps({
                    "question": question,
                    "evidence": support_payload,
                }),
            )

            hypothesis = str(
                hypothesis_json.get("hypothesis", "")
            ).strip()

        except Exception as exc:
            print(f"[NEXSUS] Hypothesis generation unavailable: {exc}")

            # Deterministic evidence-grounded fallback.
            top = supporting[0]

            hypothesis = (
                f"The available source evidence indicates that "
                f"{top['text'][:500].strip()}"
            )

        if not hypothesis:
            hypothesis = (
                "The available evidence is insufficient to establish "
                "a reliable hypothesis."
            )

        # ---------------------------------------------------------
        # 4. CHALLENGE
        # ---------------------------------------------------------
        update_stage(investigation_id, "CHALLENGE")

        challenge_query = ""

        try:
            challenge_query_json = model_json(
                client,
                "Write one semantic search query that deliberately seeks "
                "evidence contradicting, weakening, limiting, or qualifying "
                "the stated hypothesis. Return JSON {query:string}. "
                "Do not restate the hypothesis as support.",
                json.dumps({
                    "question": question,
                    "hypothesis": hypothesis,
                }),
            )

            challenge_query = str(
                challenge_query_json.get("query", "")
            ).strip()

        except Exception as exc:
            print(
                f"[NEXSUS] Challenge generation unavailable: {exc}"
            )

            challenge_query = (
                f"evidence contradicting or limiting: {hypothesis}"
            )

        counter = (
            retrieve(challenge_query)
            if challenge_query
            else []
        )

        counter = [
            item for item in counter
            if item["relevance"] >= 0.15
        ]

        all_items = {
            item["id"]: item
            for item in supporting + counter
        }

        # ---------------------------------------------------------
        # 5. VERIFY
        # ---------------------------------------------------------
        update_stage(investigation_id, "VERIFY")

        verification_json = None

        try:
            verification_json = model_json(
                client,
                "Verify the hypothesis strictly against supplied source "
                "excerpts. Return JSON {status: one of SUPPORTED, "
                "PARTIALLY_SUPPORTED, CONTRADICTED, UNVERIFIED, "
                "supporting_chunk_ids:[], counter_chunk_ids:[], "
                "limitations:[], reasoning_summary:string}. "
                "Cite only supplied chunk IDs. The summary must be concise "
                "and evidence-grounded; do not reveal chain-of-thought. "
                "If evidence is absent, status must be UNVERIFIED.",
                json.dumps({
                    "question": question,
                    "hypothesis": hypothesis,
                    "supporting_search_results": support_payload,
                    "challenge_search_results": [
                        {
                            "chunk_id": item["id"],
                            "document": item["document"],
                            "page": item["page"],
                            "text": item["text"],
                            "relevance": item["relevance"],
                        }
                        for item in counter
                    ],
                }),
            )

        except Exception as exc:
            print(f"[NEXSUS] Verification generation unavailable: {exc}")

            # -----------------------------------------------------
            # Deterministic verification fallback
            # -----------------------------------------------------
            #
            # This does NOT pretend that an LLM performed verification.
            # It uses retrieved evidence balance to produce a conservative
            # result when the LLM service is unavailable.
            #
            support_count = len(supporting)
            counter_count = len(counter)

            if support_count == 0 and counter_count == 0:
                fallback_status = "UNVERIFIED"
            elif counter_count > support_count:
                fallback_status = "CONTRADICTED"
            elif counter_count > 0:
                fallback_status = "PARTIALLY_SUPPORTED"
            else:
                fallback_status = "SUPPORTED"

            verification_json = {
                "status": fallback_status,
                "supporting_chunk_ids": [
                    item["id"] for item in supporting
                ],
                "counter_chunk_ids": [
                    item["id"] for item in counter
                ],
                "limitations": [
                    "LLM verification was unavailable; "
                    "status was determined from retrieved evidence balance."
                ],
                "reasoning_summary": (
                    "Deterministic verification fallback used because "
                    "the reasoning model was temporarily unavailable. "
                    f"Retrieved {support_count} supporting evidence item(s) "
                    f"and {counter_count} counter-evidence item(s)."
                ),
            }

        # ---------------------------------------------------------
        # 6. VALIDATE VERIFICATION
        # ---------------------------------------------------------
        status = verification_json.get("status")

        valid_statuses = {
            "SUPPORTED",
            "PARTIALLY_SUPPORTED",
            "CONTRADICTED",
            "UNVERIFIED",
        }

        if status not in valid_statuses:
            raise ValueError(
                "Verification returned an unsupported status."
            )

        support_ids = set(
            verification_json.get("supporting_chunk_ids", [])
        )

        counter_ids = set(
            verification_json.get("counter_chunk_ids", [])
        )

        supporting_ids = {
            item["id"] for item in supporting
        }

        counter_ids_available = {
            item["id"] for item in counter
        }

        if not support_ids <= supporting_ids:
            raise ValueError(
                "Verification referenced supporting evidence "
                "that was not retrieved."
            )

        if not counter_ids <= counter_ids_available:
            raise ValueError(
                "Verification referenced counter evidence "
                "that was not retrieved."
            )

        if not all_items:
            status = "UNVERIFIED"

        summary = str(
            verification_json.get("reasoning_summary", "")
        ).strip()

        if not summary:
            summary = (
                "Insufficient evidence found in the uploaded "
                "knowledge base."
            )

        strength = [
            all_items[item_id]["relevance"]
            for item_id in support_ids | counter_ids
            if item_id in all_items
        ]

        balance = (
            len(support_ids)
            / max(
                1,
                len(support_ids) + len(counter_ids),
            )
        )

        confidence = round(
            min(
                0.95,
                (
                    sum(strength) / len(strength)
                    if strength
                    else 0
                )
                * (0.5 + abs(balance - 0.5)),
            ),
            2,
        )

        # ---------------------------------------------------------
        # 7. SAVE INVESTIGATION EVIDENCE
        # ---------------------------------------------------------
        with database() as db:

            for item in supporting:
                db.execute(
                    "INSERT OR REPLACE INTO investigation_evidence "
                    "VALUES(?,?,?,?)",
                    (
                        investigation_id,
                        item["id"],
                        "supporting",
                        item["relevance"],
                    ),
                )

            for item in counter:
                db.execute(
                    "INSERT OR REPLACE INTO investigation_evidence "
                    "VALUES(?,?,?,?)",
                    (
                        investigation_id,
                        item["id"],
                        "counter",
                        item["relevance"],
                    ),
                )

            db.execute(
                "UPDATE investigations "
                "SET hypothesis=?,status=?,confidence=?,"
                "reasoning_summary=? WHERE id=?",
                (
                    hypothesis,
                    status,
                    confidence,
                    summary,
                    investigation_id,
                ),
            )

        # ---------------------------------------------------------
        # 8. UPDATE KNOWLEDGE GRAPH
        # ---------------------------------------------------------
        update_stage(investigation_id, "UPDATE")

        if len(concepts) >= 2 and status != "UNVERIFIED":

            with database() as db:

                first = db.execute(
                    "SELECT id FROM concepts "
                    "WHERE name=? COLLATE NOCASE",
                    (concepts[0],),
                ).fetchone()

                second = db.execute(
                    "SELECT id FROM concepts "
                    "WHERE name=? COLLATE NOCASE",
                    (concepts[1],),
                ).fetchone()

                if first and second:

                    if status == "CONTRADICTED":

                        db.execute(
                            "UPDATE relationships "
                            "SET status='CONTRADICTED',confidence=?,"
                            "claim=?,updated_at=? "
                            "WHERE source_id=? AND target_id=?",
                            (
                                confidence,
                                hypothesis,
                                utc_now(),
                                first["id"],
                                second["id"],
                            ),
                        )

                    else:

                        rel_id = str(uuid.uuid4())

                        graph_status = (
                            "SUPPORTED"
                            if status == "SUPPORTED"
                            else "PARTIALLY_SUPPORTED"
                        )

                        db.execute(
                            "INSERT INTO relationships("
                            "id,source_id,target_id,relationship_type,"
                            "confidence,status,claim,created_at,updated_at"
                            ") VALUES(?,?,?,?,?,?,?,?,?) "
                            "ON CONFLICT("
                            "source_id,target_id,relationship_type"
                            ") DO UPDATE SET "
                            "confidence=excluded.confidence,"
                            "status=excluded.status,"
                            "claim=excluded.claim,"
                            "updated_at=excluded.updated_at",
                            (
                                rel_id,
                                first["id"],
                                second["id"],
                                "investigated relationship",
                                confidence,
                                graph_status,
                                hypothesis,
                                utc_now(),
                                utc_now(),
                            ),
                        )

                        saved = db.execute(
                            "SELECT id FROM relationships "
                            "WHERE source_id=? AND target_id=? "
                            "AND relationship_type="
                            "'investigated relationship'",
                            (
                                first["id"],
                                second["id"],
                            ),
                        ).fetchone()

                        for chunk_id in (
                            support_ids | counter_ids
                        ):
                            db.execute(
                                "INSERT OR IGNORE INTO relationship_evidence "
                                "VALUES(?,?)",
                                (
                                    saved["id"],
                                    chunk_id,
                                ),
                            )

            with database() as db:
                db.execute(
                    "UPDATE gaps SET status='RESOLVED' WHERE id=?",
                    (inv["gap_id"],),
                )

        # ---------------------------------------------------------
        # 9. DETECT NEXT GAPS
        # ---------------------------------------------------------
        detect_gaps()

        # ---------------------------------------------------------
        # 10. COMPLETE
        # ---------------------------------------------------------
        with database() as db:
            db.execute(
                "UPDATE investigations "
                "SET status=?,stage='COMPLETE',completed_at=? "
                "WHERE id=?",
                (
                    status,
                    utc_now(),
                    investigation_id,
                ),
            )

    except Exception as exc:

        print(
            f"[NEXSUS] Investigation {investigation_id} failed: {exc}"
        )

        with database() as db:
            db.execute(
                "UPDATE investigations "
                "SET status='FAILED',stage='FAILED',"
                "reasoning_summary=? WHERE id=?",
                (
                    str(exc)[:1000],
                    investigation_id,
                ),
            )
    try:
        client = require_llm()

        with database() as db:
            inv = db.execute(
                "SELECT i.*,g.description,g.related_concepts,g.reason "
                "FROM investigations i JOIN gaps g ON g.id=i.gap_id "
                "WHERE i.id=?",
                (investigation_id,),
            ).fetchone()

        if not inv:
            return

        concepts = json.loads(inv["related_concepts"])

        # ---------------------------------------------------------
        # 1. QUESTION
        # ---------------------------------------------------------
        update_stage(investigation_id, "QUESTION")

        question = ""

        try:
            question_json = model_json(
                client,
                "Generate one focused research question for this corpus gap. "
                "Return JSON {question:string}. Do not assert facts.",
                json.dumps({
                    "gap": inv["description"],
                    "reason": inv["reason"],
                    "concepts": concepts,
                }),
            )

            question = str(
                question_json.get("question", "")
            ).strip()

        except Exception as exc:
            print(f"[NEXSUS] Question generation unavailable: {exc}")

            # Deterministic fallback
            question = (
                f"What evidence in the knowledge base explains or clarifies "
                f"the gap: {inv['description']}?"
            )

        if not question:
            raise ValueError("Could not generate a research question.")

        with database() as db:
            db.execute(
                "UPDATE investigations SET question=? WHERE id=?",
                (question, investigation_id),
            )

        # ---------------------------------------------------------
        # 2. RETRIEVE
        # ---------------------------------------------------------
        update_stage(investigation_id, "RETRIEVE")

        supporting = retrieve(
    question,
    document_id=inv["document_id"],
)

        if not supporting:
            raise ValueError(
                "No document chunks are available for retrieval."
            )

        supporting = [
            item for item in supporting
            if item["relevance"] >= 0.15
        ]

        if not supporting:
            raise ValueError(
                "No sufficiently relevant document evidence was retrieved."
            )

        support_payload = [
            {
                "chunk_id": item["id"],
                "document": item["document"],
                "page": item["page"],
                "text": item["text"],
                "relevance": item["relevance"],
            }
            for item in supporting
        ]

        # ---------------------------------------------------------
        # 3. HYPOTHESIS
        # ---------------------------------------------------------
        update_stage(investigation_id, "HYPOTHESIS")

        hypothesis = ""

        try:
            hypothesis_json = model_json(
                client,
                "Form a concise hypothesis using only supplied source "
                "excerpts. Return JSON {hypothesis:string}. "
                "If the excerpts do not support a hypothesis, state that "
                "explicitly. Do not add outside knowledge or citations.",
                json.dumps({
                    "question": question,
                    "evidence": support_payload,
                }),
            )

            hypothesis = str(
                hypothesis_json.get("hypothesis", "")
            ).strip()

        except Exception as exc:
            print(f"[NEXSUS] Hypothesis generation unavailable: {exc}")

            # Deterministic evidence-grounded fallback.
            top = supporting[0]

            hypothesis = (
                f"The available source evidence indicates that "
                f"{top['text'][:500].strip()}"
            )

        if not hypothesis:
            hypothesis = (
                "The available evidence is insufficient to establish "
                "a reliable hypothesis."
            )

        # ---------------------------------------------------------
        # 4. CHALLENGE
        # ---------------------------------------------------------
        update_stage(investigation_id, "CHALLENGE")

        challenge_query = ""

        try:
            challenge_query_json = model_json(
                client,
                "Write one semantic search query that deliberately seeks "
                "evidence contradicting, weakening, limiting, or qualifying "
                "the stated hypothesis. Return JSON {query:string}. "
                "Do not restate the hypothesis as support.",
                json.dumps({
                    "question": question,
                    "hypothesis": hypothesis,
                }),
            )

            challenge_query = str(
                challenge_query_json.get("query", "")
            ).strip()

        except Exception as exc:
            print(
                f"[NEXSUS] Challenge generation unavailable: {exc}"
            )

            challenge_query = (
                f"evidence contradicting or limiting: {hypothesis}"
            )

        counter = (
            retrieve(challenge_query)
            if challenge_query
            else []
        )

        counter = [
            item for item in counter
            if item["relevance"] >= 0.15
        ]

        all_items = {
            item["id"]: item
            for item in supporting + counter
        }

        # ---------------------------------------------------------
        # 5. VERIFY
        # ---------------------------------------------------------
        update_stage(investigation_id, "VERIFY")

        verification_json = None

        try:
            verification_json = model_json(
                client,
                "Verify the hypothesis strictly against supplied source "
                "excerpts. Return JSON {status: one of SUPPORTED, "
                "PARTIALLY_SUPPORTED, CONTRADICTED, UNVERIFIED, "
                "supporting_chunk_ids:[], counter_chunk_ids:[], "
                "limitations:[], reasoning_summary:string}. "
                "Cite only supplied chunk IDs. The summary must be concise "
                "and evidence-grounded; do not reveal chain-of-thought. "
                "If evidence is absent, status must be UNVERIFIED.",
                json.dumps({
                    "question": question,
                    "hypothesis": hypothesis,
                    "supporting_search_results": support_payload,
                    "challenge_search_results": [
                        {
                            "chunk_id": item["id"],
                            "document": item["document"],
                            "page": item["page"],
                            "text": item["text"],
                            "relevance": item["relevance"],
                        }
                        for item in counter
                    ],
                }),
            )

        except Exception as exc:
            print(f"[NEXSUS] Verification generation unavailable: {exc}")

            # -----------------------------------------------------
            # Deterministic verification fallback
            # -----------------------------------------------------
            #
            # This does NOT pretend that an LLM performed verification.
            # It uses retrieved evidence balance to produce a conservative
            # result when the LLM service is unavailable.
            #
            support_count = len(supporting)
            counter_count = len(counter)

            if support_count == 0 and counter_count == 0:
                fallback_status = "UNVERIFIED"
            elif counter_count > support_count:
                fallback_status = "CONTRADICTED"
            elif counter_count > 0:
                fallback_status = "PARTIALLY_SUPPORTED"
            else:
                fallback_status = "SUPPORTED"

            verification_json = {
                "status": fallback_status,
                "supporting_chunk_ids": [
                    item["id"] for item in supporting
                ],
                "counter_chunk_ids": [
                    item["id"] for item in counter
                ],
                "limitations": [
                    "LLM verification was unavailable; "
                    "status was determined from retrieved evidence balance."
                ],
                "reasoning_summary": (
                    "Deterministic verification fallback used because "
                    "the reasoning model was temporarily unavailable. "
                    f"Retrieved {support_count} supporting evidence item(s) "
                    f"and {counter_count} counter-evidence item(s)."
                ),
            }

        # ---------------------------------------------------------
        # 6. VALIDATE VERIFICATION
        # ---------------------------------------------------------
        status = verification_json.get("status")

        valid_statuses = {
            "SUPPORTED",
            "PARTIALLY_SUPPORTED",
            "CONTRADICTED",
            "UNVERIFIED",
        }

        if status not in valid_statuses:
            raise ValueError(
                "Verification returned an unsupported status."
            )

        support_ids = set(
            verification_json.get("supporting_chunk_ids", [])
        )

        counter_ids = set(
            verification_json.get("counter_chunk_ids", [])
        )

        supporting_ids = {
            item["id"] for item in supporting
        }

        counter_ids_available = {
            item["id"] for item in counter
        }

        if not support_ids <= supporting_ids:
            raise ValueError(
                "Verification referenced supporting evidence "
                "that was not retrieved."
            )

        if not counter_ids <= counter_ids_available:
            raise ValueError(
                "Verification referenced counter evidence "
                "that was not retrieved."
            )

        if not all_items:
            status = "UNVERIFIED"

        summary = str(
            verification_json.get("reasoning_summary", "")
        ).strip()

        if not summary:
            summary = (
                "Insufficient evidence found in the uploaded "
                "knowledge base."
            )

        strength = [
            all_items[item_id]["relevance"]
            for item_id in support_ids | counter_ids
            if item_id in all_items
        ]

        balance = (
            len(support_ids)
            / max(
                1,
                len(support_ids) + len(counter_ids),
            )
        )

        confidence = round(
            min(
                0.95,
                (
                    sum(strength) / len(strength)
                    if strength
                    else 0
                )
                * (0.5 + abs(balance - 0.5)),
            ),
            2,
        )

        # ---------------------------------------------------------
        # 7. SAVE INVESTIGATION EVIDENCE
        # ---------------------------------------------------------
        with database() as db:

            for item in supporting:
                db.execute(
                    "INSERT OR REPLACE INTO investigation_evidence "
                    "VALUES(?,?,?,?)",
                    (
                        investigation_id,
                        item["id"],
                        "supporting",
                        item["relevance"],
                    ),
                )

            for item in counter:
                db.execute(
                    "INSERT OR REPLACE INTO investigation_evidence "
                    "VALUES(?,?,?,?)",
                    (
                        investigation_id,
                        item["id"],
                        "counter",
                        item["relevance"],
                    ),
                )

            db.execute(
                "UPDATE investigations "
                "SET hypothesis=?,status=?,confidence=?,"
                "reasoning_summary=? WHERE id=?",
                (
                    hypothesis,
                    status,
                    confidence,
                    summary,
                    investigation_id,
                ),
            )

        # ---------------------------------------------------------
        # 8. UPDATE KNOWLEDGE GRAPH
        # ---------------------------------------------------------
        update_stage(investigation_id, "UPDATE")

        if len(concepts) >= 2 and status != "UNVERIFIED":

            with database() as db:

                first = db.execute(
                    "SELECT id FROM concepts "
                    "WHERE name=? COLLATE NOCASE",
                    (concepts[0],),
                ).fetchone()

                second = db.execute(
                    "SELECT id FROM concepts "
                    "WHERE name=? COLLATE NOCASE",
                    (concepts[1],),
                ).fetchone()

                if first and second:

                    if status == "CONTRADICTED":

                        db.execute(
                            "UPDATE relationships "
                            "SET status='CONTRADICTED',confidence=?,"
                            "claim=?,updated_at=? "
                            "WHERE source_id=? AND target_id=?",
                            (
                                confidence,
                                hypothesis,
                                utc_now(),
                                first["id"],
                                second["id"],
                            ),
                        )

                    else:

                        rel_id = str(uuid.uuid4())

                        graph_status = (
                            "SUPPORTED"
                            if status == "SUPPORTED"
                            else "PARTIALLY_SUPPORTED"
                        )

                        db.execute(
                            "INSERT INTO relationships("
                            "id,source_id,target_id,relationship_type,"
                            "confidence,status,claim,created_at,updated_at"
                            ") VALUES(?,?,?,?,?,?,?,?,?) "
                            "ON CONFLICT("
                            "source_id,target_id,relationship_type"
                            ") DO UPDATE SET "
                            "confidence=excluded.confidence,"
                            "status=excluded.status,"
                            "claim=excluded.claim,"
                            "updated_at=excluded.updated_at",
                            (
                                rel_id,
                                first["id"],
                                second["id"],
                                "investigated relationship",
                                confidence,
                                graph_status,
                                hypothesis,
                                utc_now(),
                                utc_now(),
                            ),
                        )

                        saved = db.execute(
                            "SELECT id FROM relationships "
                            "WHERE source_id=? AND target_id=? "
                            "AND relationship_type="
                            "'investigated relationship'",
                            (
                                first["id"],
                                second["id"],
                            ),
                        ).fetchone()

                        for chunk_id in (
                            support_ids | counter_ids
                        ):
                            db.execute(
                                "INSERT OR IGNORE INTO relationship_evidence "
                                "VALUES(?,?)",
                                (
                                    saved["id"],
                                    chunk_id,
                                ),
                            )

            with database() as db:
                db.execute(
                    "UPDATE gaps SET status='RESOLVED' WHERE id=?",
                    (inv["gap_id"],),
                )

        # ---------------------------------------------------------
        # 9. DETECT NEXT GAPS
        # ---------------------------------------------------------
        detect_gaps()

        # ---------------------------------------------------------
        # 10. COMPLETE
        # ---------------------------------------------------------
        with database() as db:
            db.execute(
                "UPDATE investigations "
                "SET status=?,stage='COMPLETE',completed_at=? "
                "WHERE id=?",
                (
                    status,
                    utc_now(),
                    investigation_id,
                ),
            )

    except Exception as exc:

        print(
            f"[NEXSUS] Investigation {investigation_id} failed: {exc}"
        )

        with database() as db:
            db.execute(
                "UPDATE investigations "
                "SET status='FAILED',stage='FAILED',"
                "reasoning_summary=? WHERE id=?",
                (
                    str(exc)[:1000],
                    investigation_id,
                ),
            )
    try:
        client = require_llm()

        with database() as db:
            inv = db.execute(
                "SELECT i.*,g.description,g.related_concepts,g.reason "
                "FROM investigations i JOIN gaps g ON g.id=i.gap_id "
                "WHERE i.id=?",
                (investigation_id,),
            ).fetchone()

        if not inv:
            return

        concepts = json.loads(inv["related_concepts"])

        # ---------------------------------------------------------
        # 1. QUESTION
        # ---------------------------------------------------------
        update_stage(investigation_id, "QUESTION")

        question = ""

        try:
            question_json = model_json(
                client,
                "Generate one focused research question for this corpus gap. "
                "Return JSON {question:string}. Do not assert facts.",
                json.dumps({
                    "gap": inv["description"],
                    "reason": inv["reason"],
                    "concepts": concepts,
                }),
            )

            question = str(
                question_json.get("question", "")
            ).strip()

        except Exception as exc:
            print(f"[NEXSUS] Question generation unavailable: {exc}")

            # Deterministic fallback
            question = (
                f"What evidence in the knowledge base explains or clarifies "
                f"the gap: {inv['description']}?"
            )

        if not question:
            raise ValueError("Could not generate a research question.")

        with database() as db:
            db.execute(
                "UPDATE investigations SET question=? WHERE id=?",
                (question, investigation_id),
            )

        # ---------------------------------------------------------
        # 2. RETRIEVE
        # ---------------------------------------------------------
        update_stage(investigation_id, "RETRIEVE")

        supporting = retrieve(
    question,
    document_id=inv["document_id"],
)

        if not supporting:
            raise ValueError(
                "No document chunks are available for retrieval."
            )

        supporting = [
            item for item in supporting
            if item["relevance"] >= 0.15
        ]

        if not supporting:
            raise ValueError(
                "No sufficiently relevant document evidence was retrieved."
            )

        support_payload = [
            {
                "chunk_id": item["id"],
                "document": item["document"],
                "page": item["page"],
                "text": item["text"],
                "relevance": item["relevance"],
            }
            for item in supporting
        ]

        # ---------------------------------------------------------
        # 3. HYPOTHESIS
        # ---------------------------------------------------------
        update_stage(investigation_id, "HYPOTHESIS")

        hypothesis = ""

        try:
            hypothesis_json = model_json(
                client,
                "Form a concise hypothesis using only supplied source "
                "excerpts. Return JSON {hypothesis:string}. "
                "If the excerpts do not support a hypothesis, state that "
                "explicitly. Do not add outside knowledge or citations.",
                json.dumps({
                    "question": question,
                    "evidence": support_payload,
                }),
            )

            hypothesis = str(
                hypothesis_json.get("hypothesis", "")
            ).strip()

        except Exception as exc:
            print(f"[NEXSUS] Hypothesis generation unavailable: {exc}")

            # Deterministic evidence-grounded fallback.
            top = supporting[0]

            hypothesis = (
                f"The available source evidence indicates that "
                f"{top['text'][:500].strip()}"
            )

        if not hypothesis:
            hypothesis = (
                "The available evidence is insufficient to establish "
                "a reliable hypothesis."
            )

        # ---------------------------------------------------------
        # 4. CHALLENGE
        # ---------------------------------------------------------
        update_stage(investigation_id, "CHALLENGE")

        challenge_query = ""

        try:
            challenge_query_json = model_json(
                client,
                "Write one semantic search query that deliberately seeks "
                "evidence contradicting, weakening, limiting, or qualifying "
                "the stated hypothesis. Return JSON {query:string}. "
                "Do not restate the hypothesis as support.",
                json.dumps({
                    "question": question,
                    "hypothesis": hypothesis,
                }),
            )

            challenge_query = str(
                challenge_query_json.get("query", "")
            ).strip()

        except Exception as exc:
            print(
                f"[NEXSUS] Challenge generation unavailable: {exc}"
            )

            challenge_query = (
                f"evidence contradicting or limiting: {hypothesis}"
            )

        counter = (
            retrieve(challenge_query)
            if challenge_query
            else []
        )

        counter = [
            item for item in counter
            if item["relevance"] >= 0.15
        ]

        all_items = {
            item["id"]: item
            for item in supporting + counter
        }

        # ---------------------------------------------------------
        # 5. VERIFY
        # ---------------------------------------------------------
        update_stage(investigation_id, "VERIFY")

        verification_json = None

        try:
            verification_json = model_json(
                client,
                "Verify the hypothesis strictly against supplied source "
                "excerpts. Return JSON {status: one of SUPPORTED, "
                "PARTIALLY_SUPPORTED, CONTRADICTED, UNVERIFIED, "
                "supporting_chunk_ids:[], counter_chunk_ids:[], "
                "limitations:[], reasoning_summary:string}. "
                "Cite only supplied chunk IDs. The summary must be concise "
                "and evidence-grounded; do not reveal chain-of-thought. "
                "If evidence is absent, status must be UNVERIFIED.",
                json.dumps({
                    "question": question,
                    "hypothesis": hypothesis,
                    "supporting_search_results": support_payload,
                    "challenge_search_results": [
                        {
                            "chunk_id": item["id"],
                            "document": item["document"],
                            "page": item["page"],
                            "text": item["text"],
                            "relevance": item["relevance"],
                        }
                        for item in counter
                    ],
                }),
            )

        except Exception as exc:
            print(f"[NEXSUS] Verification generation unavailable: {exc}")

            # -----------------------------------------------------
            # Deterministic verification fallback
            # -----------------------------------------------------
            #
            # This does NOT pretend that an LLM performed verification.
            # It uses retrieved evidence balance to produce a conservative
            # result when the LLM service is unavailable.
            #
            support_count = len(supporting)
            counter_count = len(counter)

            if support_count == 0 and counter_count == 0:
                fallback_status = "UNVERIFIED"
            elif counter_count > support_count:
                fallback_status = "CONTRADICTED"
            elif counter_count > 0:
                fallback_status = "PARTIALLY_SUPPORTED"
            else:
                fallback_status = "SUPPORTED"

            verification_json = {
                "status": fallback_status,
                "supporting_chunk_ids": [
                    item["id"] for item in supporting
                ],
                "counter_chunk_ids": [
                    item["id"] for item in counter
                ],
                "limitations": [
                    "LLM verification was unavailable; "
                    "status was determined from retrieved evidence balance."
                ],
                "reasoning_summary": (
                    "Deterministic verification fallback used because "
                    "the reasoning model was temporarily unavailable. "
                    f"Retrieved {support_count} supporting evidence item(s) "
                    f"and {counter_count} counter-evidence item(s)."
                ),
            }

        # ---------------------------------------------------------
        # 6. VALIDATE VERIFICATION
        # ---------------------------------------------------------
        status = verification_json.get("status")

        valid_statuses = {
            "SUPPORTED",
            "PARTIALLY_SUPPORTED",
            "CONTRADICTED",
            "UNVERIFIED",
        }

        if status not in valid_statuses:
            raise ValueError(
                "Verification returned an unsupported status."
            )

        support_ids = set(
            verification_json.get("supporting_chunk_ids", [])
        )

        counter_ids = set(
            verification_json.get("counter_chunk_ids", [])
        )

        supporting_ids = {
            item["id"] for item in supporting
        }

        counter_ids_available = {
            item["id"] for item in counter
        }

        if not support_ids <= supporting_ids:
            raise ValueError(
                "Verification referenced supporting evidence "
                "that was not retrieved."
            )

        if not counter_ids <= counter_ids_available:
            raise ValueError(
                "Verification referenced counter evidence "
                "that was not retrieved."
            )

        if not all_items:
            status = "UNVERIFIED"

        summary = str(
            verification_json.get("reasoning_summary", "")
        ).strip()

        if not summary:
            summary = (
                "Insufficient evidence found in the uploaded "
                "knowledge base."
            )

        strength = [
            all_items[item_id]["relevance"]
            for item_id in support_ids | counter_ids
            if item_id in all_items
        ]

        balance = (
            len(support_ids)
            / max(
                1,
                len(support_ids) + len(counter_ids),
            )
        )

        confidence = round(
            min(
                0.95,
                (
                    sum(strength) / len(strength)
                    if strength
                    else 0
                )
                * (0.5 + abs(balance - 0.5)),
            ),
            2,
        )

        # ---------------------------------------------------------
        # 7. SAVE INVESTIGATION EVIDENCE
        # ---------------------------------------------------------
        with database() as db:

            for item in supporting:
                db.execute(
                    "INSERT OR REPLACE INTO investigation_evidence "
                    "VALUES(?,?,?,?)",
                    (
                        investigation_id,
                        item["id"],
                        "supporting",
                        item["relevance"],
                    ),
                )

            for item in counter:
                db.execute(
                    "INSERT OR REPLACE INTO investigation_evidence "
                    "VALUES(?,?,?,?)",
                    (
                        investigation_id,
                        item["id"],
                        "counter",
                        item["relevance"],
                    ),
                )

            db.execute(
                "UPDATE investigations "
                "SET hypothesis=?,status=?,confidence=?,"
                "reasoning_summary=? WHERE id=?",
                (
                    hypothesis,
                    status,
                    confidence,
                    summary,
                    investigation_id,
                ),
            )

        # ---------------------------------------------------------
        # 8. UPDATE KNOWLEDGE GRAPH
        # ---------------------------------------------------------
        update_stage(investigation_id, "UPDATE")

        if len(concepts) >= 2 and status != "UNVERIFIED":

            with database() as db:

                first = db.execute(
                    "SELECT id FROM concepts "
                    "WHERE name=? COLLATE NOCASE",
                    (concepts[0],),
                ).fetchone()

                second = db.execute(
                    "SELECT id FROM concepts "
                    "WHERE name=? COLLATE NOCASE",
                    (concepts[1],),
                ).fetchone()

                if first and second:

                    if status == "CONTRADICTED":

                        db.execute(
                            "UPDATE relationships "
                            "SET status='CONTRADICTED',confidence=?,"
                            "claim=?,updated_at=? "
                            "WHERE source_id=? AND target_id=?",
                            (
                                confidence,
                                hypothesis,
                                utc_now(),
                                first["id"],
                                second["id"],
                            ),
                        )

                    else:

                        rel_id = str(uuid.uuid4())

                        graph_status = (
                            "SUPPORTED"
                            if status == "SUPPORTED"
                            else "PARTIALLY_SUPPORTED"
                        )

                        db.execute(
                            "INSERT INTO relationships("
                            "id,source_id,target_id,relationship_type,"
                            "confidence,status,claim,created_at,updated_at"
                            ") VALUES(?,?,?,?,?,?,?,?,?) "
                            "ON CONFLICT("
                            "source_id,target_id,relationship_type"
                            ") DO UPDATE SET "
                            "confidence=excluded.confidence,"
                            "status=excluded.status,"
                            "claim=excluded.claim,"
                            "updated_at=excluded.updated_at",
                            (
                                rel_id,
                                first["id"],
                                second["id"],
                                "investigated relationship",
                                confidence,
                                graph_status,
                                hypothesis,
                                utc_now(),
                                utc_now(),
                            ),
                        )

                        saved = db.execute(
                            "SELECT id FROM relationships "
                            "WHERE source_id=? AND target_id=? "
                            "AND relationship_type="
                            "'investigated relationship'",
                            (
                                first["id"],
                                second["id"],
                            ),
                        ).fetchone()

                        for chunk_id in (
                            support_ids | counter_ids
                        ):
                            db.execute(
                                "INSERT OR IGNORE INTO relationship_evidence "
                                "VALUES(?,?)",
                                (
                                    saved["id"],
                                    chunk_id,
                                ),
                            )

            with database() as db:
                db.execute(
                    "UPDATE gaps SET status='RESOLVED' WHERE id=?",
                    (inv["gap_id"],),
                )

        # ---------------------------------------------------------
        # 9. DETECT NEXT GAPS
        # ---------------------------------------------------------
        detect_gaps()

        # ---------------------------------------------------------
        # 10. COMPLETE
        # ---------------------------------------------------------
        with database() as db:
            db.execute(
                "UPDATE investigations "
                "SET status=?,stage='COMPLETE',completed_at=? "
                "WHERE id=?",
                (
                    status,
                    utc_now(),
                    investigation_id,
                ),
            )

    except Exception as exc:

        print(
            f"[NEXSUS] Investigation {investigation_id} failed: {exc}"
        )

        with database() as db:
            db.execute(
                "UPDATE investigations "
                "SET status='FAILED',stage='FAILED',"
                "reasoning_summary=? WHERE id=?",
                (
                    str(exc)[:1000],
                    investigation_id,
                ),
            )
    try:
        client = require_llm()

        with database() as db:
            inv = db.execute(
                "SELECT i.*,g.description,g.related_concepts,g.reason "
                "FROM investigations i JOIN gaps g ON g.id=i.gap_id "
                "WHERE i.id=?",
                (investigation_id,),
            ).fetchone()

        if not inv:
            return

        concepts = json.loads(inv["related_concepts"])

        # ---------------------------------------------------------
        # 1. QUESTION
        # ---------------------------------------------------------
        update_stage(investigation_id, "QUESTION")

        question = ""

        try:
            question_json = model_json(
                client,
                "Generate one focused research question for this corpus gap. "
                "Return JSON {question:string}. Do not assert facts.",
                json.dumps({
                    "gap": inv["description"],
                    "reason": inv["reason"],
                    "concepts": concepts,
                }),
            )

            question = str(
                question_json.get("question", "")
            ).strip()

        except Exception as exc:
            print(f"[NEXSUS] Question generation unavailable: {exc}")

            # Deterministic fallback
            question = (
                f"What evidence in the knowledge base explains or clarifies "
                f"the gap: {inv['description']}?"
            )

        if not question:
            raise ValueError("Could not generate a research question.")

        with database() as db:
            db.execute(
                "UPDATE investigations SET question=? WHERE id=?",
                (question, investigation_id),
            )

        # ---------------------------------------------------------
        # 2. RETRIEVE
        # ---------------------------------------------------------
        update_stage(investigation_id, "RETRIEVE")

        supporting = retrieve(
    question,
    document_id=inv["document_id"],
)

        if not supporting:
            raise ValueError(
                "No document chunks are available for retrieval."
            )

        supporting = [
            item for item in supporting
            if item["relevance"] >= 0.15
        ]

        if not supporting:
            raise ValueError(
                "No sufficiently relevant document evidence was retrieved."
            )

        support_payload = [
            {
                "chunk_id": item["id"],
                "document": item["document"],
                "page": item["page"],
                "text": item["text"],
                "relevance": item["relevance"],
            }
            for item in supporting
        ]

        # ---------------------------------------------------------
        # 3. HYPOTHESIS
        # ---------------------------------------------------------
        update_stage(investigation_id, "HYPOTHESIS")

        hypothesis = ""

        try:
            hypothesis_json = model_json(
                client,
                "Form a concise hypothesis using only supplied source "
                "excerpts. Return JSON {hypothesis:string}. "
                "If the excerpts do not support a hypothesis, state that "
                "explicitly. Do not add outside knowledge or citations.",
                json.dumps({
                    "question": question,
                    "evidence": support_payload,
                }),
            )

            hypothesis = str(
                hypothesis_json.get("hypothesis", "")
            ).strip()

        except Exception as exc:
            print(f"[NEXSUS] Hypothesis generation unavailable: {exc}")

            # Deterministic evidence-grounded fallback.
            top = supporting[0]

            hypothesis = (
                f"The available source evidence indicates that "
                f"{top['text'][:500].strip()}"
            )

        if not hypothesis:
            hypothesis = (
                "The available evidence is insufficient to establish "
                "a reliable hypothesis."
            )

        # ---------------------------------------------------------
        # 4. CHALLENGE
        # ---------------------------------------------------------
        update_stage(investigation_id, "CHALLENGE")

        challenge_query = ""

        try:
            challenge_query_json = model_json(
                client,
                "Write one semantic search query that deliberately seeks "
                "evidence contradicting, weakening, limiting, or qualifying "
                "the stated hypothesis. Return JSON {query:string}. "
                "Do not restate the hypothesis as support.",
                json.dumps({
                    "question": question,
                    "hypothesis": hypothesis,
                }),
            )

            challenge_query = str(
                challenge_query_json.get("query", "")
            ).strip()

        except Exception as exc:
            print(
                f"[NEXSUS] Challenge generation unavailable: {exc}"
            )

            challenge_query = (
                f"evidence contradicting or limiting: {hypothesis}"
            )

        counter = (
            retrieve(challenge_query)
            if challenge_query
            else []
        )

        counter = [
            item for item in counter
            if item["relevance"] >= 0.15
        ]

        all_items = {
            item["id"]: item
            for item in supporting + counter
        }

        # ---------------------------------------------------------
        # 5. VERIFY
        # ---------------------------------------------------------
        update_stage(investigation_id, "VERIFY")

        verification_json = None

        try:
            verification_json = model_json(
                client,
                "Verify the hypothesis strictly against supplied source "
                "excerpts. Return JSON {status: one of SUPPORTED, "
                "PARTIALLY_SUPPORTED, CONTRADICTED, UNVERIFIED, "
                "supporting_chunk_ids:[], counter_chunk_ids:[], "
                "limitations:[], reasoning_summary:string}. "
                "Cite only supplied chunk IDs. The summary must be concise "
                "and evidence-grounded; do not reveal chain-of-thought. "
                "If evidence is absent, status must be UNVERIFIED.",
                json.dumps({
                    "question": question,
                    "hypothesis": hypothesis,
                    "supporting_search_results": support_payload,
                    "challenge_search_results": [
                        {
                            "chunk_id": item["id"],
                            "document": item["document"],
                            "page": item["page"],
                            "text": item["text"],
                            "relevance": item["relevance"],
                        }
                        for item in counter
                    ],
                }),
            )

        except Exception as exc:
            print(f"[NEXSUS] Verification generation unavailable: {exc}")

            # -----------------------------------------------------
            # Deterministic verification fallback
            # -----------------------------------------------------
            #
            # This does NOT pretend that an LLM performed verification.
            # It uses retrieved evidence balance to produce a conservative
            # result when the LLM service is unavailable.
            #
            support_count = len(supporting)
            counter_count = len(counter)

            if support_count == 0 and counter_count == 0:
                fallback_status = "UNVERIFIED"
            elif counter_count > support_count:
                fallback_status = "CONTRADICTED"
            elif counter_count > 0:
                fallback_status = "PARTIALLY_SUPPORTED"
            else:
                fallback_status = "SUPPORTED"

            verification_json = {
                "status": fallback_status,
                "supporting_chunk_ids": [
                    item["id"] for item in supporting
                ],
                "counter_chunk_ids": [
                    item["id"] for item in counter
                ],
                "limitations": [
                    "LLM verification was unavailable; "
                    "status was determined from retrieved evidence balance."
                ],
                "reasoning_summary": (
                    "Deterministic verification fallback used because "
                    "the reasoning model was temporarily unavailable. "
                    f"Retrieved {support_count} supporting evidence item(s) "
                    f"and {counter_count} counter-evidence item(s)."
                ),
            }

        # ---------------------------------------------------------
        # 6. VALIDATE VERIFICATION
        # ---------------------------------------------------------
        status = verification_json.get("status")

        valid_statuses = {
            "SUPPORTED",
            "PARTIALLY_SUPPORTED",
            "CONTRADICTED",
            "UNVERIFIED",
        }

        if status not in valid_statuses:
            raise ValueError(
                "Verification returned an unsupported status."
            )

        support_ids = set(
            verification_json.get("supporting_chunk_ids", [])
        )

        counter_ids = set(
            verification_json.get("counter_chunk_ids", [])
        )

        supporting_ids = {
            item["id"] for item in supporting
        }

        counter_ids_available = {
            item["id"] for item in counter
        }

        if not support_ids <= supporting_ids:
            raise ValueError(
                "Verification referenced supporting evidence "
                "that was not retrieved."
            )

        if not counter_ids <= counter_ids_available:
            raise ValueError(
                "Verification referenced counter evidence "
                "that was not retrieved."
            )

        if not all_items:
            status = "UNVERIFIED"

        summary = str(
            verification_json.get("reasoning_summary", "")
        ).strip()

        if not summary:
            summary = (
                "Insufficient evidence found in the uploaded "
                "knowledge base."
            )

        strength = [
            all_items[item_id]["relevance"]
            for item_id in support_ids | counter_ids
            if item_id in all_items
        ]

        balance = (
            len(support_ids)
            / max(
                1,
                len(support_ids) + len(counter_ids),
            )
        )

        confidence = round(
            min(
                0.95,
                (
                    sum(strength) / len(strength)
                    if strength
                    else 0
                )
                * (0.5 + abs(balance - 0.5)),
            ),
            2,
        )

        # ---------------------------------------------------------
        # 7. SAVE INVESTIGATION EVIDENCE
        # ---------------------------------------------------------
        with database() as db:

            for item in supporting:
                db.execute(
                    "INSERT OR REPLACE INTO investigation_evidence "
                    "VALUES(?,?,?,?)",
                    (
                        investigation_id,
                        item["id"],
                        "supporting",
                        item["relevance"],
                    ),
                )

            for item in counter:
                db.execute(
                    "INSERT OR REPLACE INTO investigation_evidence "
                    "VALUES(?,?,?,?)",
                    (
                        investigation_id,
                        item["id"],
                        "counter",
                        item["relevance"],
                    ),
                )

            db.execute(
                "UPDATE investigations "
                "SET hypothesis=?,status=?,confidence=?,"
                "reasoning_summary=? WHERE id=?",
                (
                    hypothesis,
                    status,
                    confidence,
                    summary,
                    investigation_id,
                ),
            )

        # ---------------------------------------------------------
        # 8. UPDATE KNOWLEDGE GRAPH
        # ---------------------------------------------------------
        update_stage(investigation_id, "UPDATE")

        if len(concepts) >= 2 and status != "UNVERIFIED":

            with database() as db:

                first = db.execute(
                    "SELECT id FROM concepts "
                    "WHERE name=? COLLATE NOCASE",
                    (concepts[0],),
                ).fetchone()

                second = db.execute(
                    "SELECT id FROM concepts "
                    "WHERE name=? COLLATE NOCASE",
                    (concepts[1],),
                ).fetchone()

                if first and second:

                    if status == "CONTRADICTED":

                        db.execute(
                            "UPDATE relationships "
                            "SET status='CONTRADICTED',confidence=?,"
                            "claim=?,updated_at=? "
                            "WHERE source_id=? AND target_id=?",
                            (
                                confidence,
                                hypothesis,
                                utc_now(),
                                first["id"],
                                second["id"],
                            ),
                        )

                    else:

                        rel_id = str(uuid.uuid4())

                        graph_status = (
                            "SUPPORTED"
                            if status == "SUPPORTED"
                            else "PARTIALLY_SUPPORTED"
                        )

                        db.execute(
                            "INSERT INTO relationships("
                            "id,source_id,target_id,relationship_type,"
                            "confidence,status,claim,created_at,updated_at"
                            ") VALUES(?,?,?,?,?,?,?,?,?) "
                            "ON CONFLICT("
                            "source_id,target_id,relationship_type"
                            ") DO UPDATE SET "
                            "confidence=excluded.confidence,"
                            "status=excluded.status,"
                            "claim=excluded.claim,"
                            "updated_at=excluded.updated_at",
                            (
                                rel_id,
                                first["id"],
                                second["id"],
                                "investigated relationship",
                                confidence,
                                graph_status,
                                hypothesis,
                                utc_now(),
                                utc_now(),
                            ),
                        )

                        saved = db.execute(
                            "SELECT id FROM relationships "
                            "WHERE source_id=? AND target_id=? "
                            "AND relationship_type="
                            "'investigated relationship'",
                            (
                                first["id"],
                                second["id"],
                            ),
                        ).fetchone()

                        for chunk_id in (
                            support_ids | counter_ids
                        ):
                            db.execute(
                                "INSERT OR IGNORE INTO relationship_evidence "
                                "VALUES(?,?)",
                                (
                                    saved["id"],
                                    chunk_id,
                                ),
                            )

            with database() as db:
                db.execute(
                    "UPDATE gaps SET status='RESOLVED' WHERE id=?",
                    (inv["gap_id"],),
                )

        # ---------------------------------------------------------
        # 9. DETECT NEXT GAPS
        # ---------------------------------------------------------
        detect_gaps()

        # ---------------------------------------------------------
        # 10. COMPLETE
        # ---------------------------------------------------------
        with database() as db:
            db.execute(
                "UPDATE investigations "
                "SET status=?,stage='COMPLETE',completed_at=? "
                "WHERE id=?",
                (
                    status,
                    utc_now(),
                    investigation_id,
                ),
            )

    except Exception as exc:

        print(
            f"[NEXSUS] Investigation {investigation_id} failed: {exc}"
        )

        with database() as db:
            db.execute(
                "UPDATE investigations "
                "SET status='FAILED',stage='FAILED',"
                "reasoning_summary=? WHERE id=?",
                (
                    str(exc)[:1000],
                    investigation_id,
                ),
            )
    try:
        client = require_llm()
        with database() as db:
           inv = db.execute(
        "SELECT i.*,g.description,g.related_concepts,g.reason "
        "FROM investigations i "
        "JOIN gaps g ON g.id=i.gap_id "
        "WHERE i.id=?",
        (investigation_id,),
    ).fetchone()
        if not inv:
            return
        concepts = json.loads(inv["related_concepts"])
        update_stage(investigation_id, "QUESTION")
        question_json = model_json(client, "Generate one focused research question for this corpus gap. Return JSON {question:string}. Do not assert facts.",
                                   json.dumps({"gap": inv["description"], "reason": inv["reason"], "concepts": concepts}))
        question = str(question_json.get("question", "")).strip()
        if not question:
            raise ValueError("The LLM did not return a research question.")
        with database() as db:
            db.execute("UPDATE investigations SET question=? WHERE id=?", (question, investigation_id))

        update_stage(investigation_id, "RETRIEVE")
        supporting = retrieve(
    question,
    document_id=inv["document_id"],
)
        if not supporting:
            raise ValueError("No document chunks are available for retrieval.")
        update_stage(investigation_id, "HYPOTHESIS")
        support_payload = [{"chunk_id": item["id"], "document": item["document"], "page": item["page"],
                            "text": item["text"], "relevance": item["relevance"]} for item in supporting]
        hypothesis_json = model_json(
            client,
            "Form a concise hypothesis using only supplied source excerpts. Return JSON {hypothesis:string}. "
            "If the excerpts do not support a hypothesis, state that explicitly. Do not add outside knowledge or citations.",
            json.dumps({"question": question, "evidence": support_payload}),
        )
        hypothesis = str(hypothesis_json.get("hypothesis", "")).strip()
        if not hypothesis:
            raise ValueError("The LLM did not return a hypothesis.")

        update_stage(investigation_id, "CHALLENGE")

        challenge_query = ""

        try:
            challenge_query_json = model_json(
                client,
                "Write one semantic search query that deliberately seeks evidence "
                "contradicting, weakening, limiting, or qualifying the stated "
                "hypothesis. Return JSON {query:string}. Do not restate the "
                "hypothesis as support.",
                json.dumps({
                    "question": question,
                    "hypothesis": hypothesis,
                }),
            )

            challenge_query = str(
                challenge_query_json.get("query", "")
            ).strip()

        except Exception as exc:
            print(
                f"[NEXSUS] Challenge generation unavailable: {exc}"
            )

            # Deterministic fallback:
            # search directly for evidence that could weaken the hypothesis.
            challenge_query = (
                f"evidence contradicting or limiting: {hypothesis}"
            )

        counter = (
    retrieve(
        challenge_query,
        document_id=inv["document_id"],
    )
    if challenge_query
    else []
)
        supporting = [item for item in supporting if item["relevance"] >= 0.15]
        counter = [item for item in counter if item["relevance"] >= 0.15]
        all_items = {item["id"]: item for item in supporting + counter}
        update_stage(investigation_id, "VERIFY")
        verification_json = model_json(
            client,
            "Verify the hypothesis strictly against supplied source excerpts. Return JSON {status: one of SUPPORTED, "
            "PARTIALLY_SUPPORTED, CONTRADICTED, UNVERIFIED, supporting_chunk_ids:[], counter_chunk_ids:[], "
            "limitations:[], reasoning_summary:string}. Cite only supplied chunk IDs. The summary must be concise "
            "and evidence-grounded; do not reveal chain-of-thought. If evidence is absent, status must be UNVERIFIED.",
            json.dumps({"question": question, "hypothesis": hypothesis,
                        "supporting_search_results": support_payload,
                        "challenge_search_results": [{"chunk_id": item["id"], "document": item["document"],
                            "page": item["page"], "text": item["text"], "relevance": item["relevance"]} for item in counter]}),
        )
        status = verification_json.get("status")
        valid_statuses = {"SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "UNVERIFIED"}
        if status not in valid_statuses:
            raise ValueError("Verification returned an unsupported status.")
        support_ids = set(verification_json.get("supporting_chunk_ids", []))
        counter_ids = set(verification_json.get("counter_chunk_ids", []))
        if not support_ids <= {item["id"] for item in supporting} or not counter_ids <= {item["id"] for item in counter}:
            raise ValueError("Verification referenced evidence that was not retrieved.")
        if not all_items:
            status = "UNVERIFIED"
        summary = str(verification_json.get("reasoning_summary", "")).strip()
        if not summary:
            summary = "Insufficient evidence found in the uploaded knowledge base."
        strength = [all_items[item_id]["relevance"] for item_id in support_ids | counter_ids if item_id in all_items]
        balance = len(support_ids) / max(1, len(support_ids) + len(counter_ids))
        confidence = round(min(0.95, (sum(strength) / len(strength) if strength else 0) * (0.5 + abs(balance - 0.5))), 2)

        with database() as db:
            for item in supporting:
                db.execute("INSERT OR REPLACE INTO investigation_evidence VALUES(?,?,?,?)",
                           (investigation_id, item["id"], "supporting", item["relevance"]))
            for item in counter:
                db.execute("INSERT OR REPLACE INTO investigation_evidence VALUES(?,?,?,?)",
                           (investigation_id, item["id"], "counter", item["relevance"]))
            db.execute("UPDATE investigations SET hypothesis=?,status=?,confidence=?,reasoning_summary=? WHERE id=?",
                       (hypothesis, status, confidence, summary, investigation_id))

        update_stage(investigation_id, "UPDATE")
        if len(concepts) >= 2 and status != "UNVERIFIED":
            with database() as db:
                first = db.execute("SELECT id FROM concepts WHERE name=? COLLATE NOCASE", (concepts[0],)).fetchone()
                second = db.execute("SELECT id FROM concepts WHERE name=? COLLATE NOCASE", (concepts[1],)).fetchone()
                if first and second:
                    if status == "CONTRADICTED":
                        db.execute("UPDATE relationships SET status='CONTRADICTED',confidence=?,claim=?,updated_at=? WHERE source_id=? AND target_id=?",
                                   (confidence, hypothesis, utc_now(), first["id"], second["id"]))
                    else:
                        rel_id = str(uuid.uuid4())
                        graph_status = "SUPPORTED" if status == "SUPPORTED" else "PARTIALLY_SUPPORTED"
                        db.execute(
                            "INSERT INTO relationships(id,source_id,target_id,relationship_type,confidence,status,claim,created_at,updated_at) "
                            "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(source_id,target_id,relationship_type) DO UPDATE SET "
                            "confidence=excluded.confidence,status=excluded.status,claim=excluded.claim,updated_at=excluded.updated_at",
                            (rel_id, first["id"], second["id"], "investigated relationship", confidence,
                             graph_status, hypothesis, utc_now(), utc_now()),
                        )
                        saved = db.execute("SELECT id FROM relationships WHERE source_id=? AND target_id=? AND relationship_type='investigated relationship'",
                                           (first["id"], second["id"])).fetchone()
                        for chunk_id in support_ids | counter_ids:
                            db.execute("INSERT OR IGNORE INTO relationship_evidence VALUES(?,?)", (saved["id"], chunk_id))
            with database() as db:
                db.execute("UPDATE gaps SET status='RESOLVED' WHERE id=?", (inv["gap_id"],))
        detect_gaps()
        with database() as db:
            db.execute("UPDATE investigations SET status=?,stage='COMPLETE',completed_at=? WHERE id=?",
                       (status, utc_now(), investigation_id))
    except Exception as exc:
        with database() as db:
            db.execute("UPDATE investigations SET status='FAILED',stage='FAILED',reasoning_summary=? WHERE id=?",
                       (str(exc)[:1000], investigation_id))


@app.get("/health")
def health() -> dict:
    provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    return {"status": "ok", "mode": "live", "llm_configured": bool(os.getenv("GEMINI_API_KEY")),
            "embedding_provider": provider,
            "embedding_dimensions": LOCAL_EMBEDDING_DIMENSIONS if provider == "local" else None}


@app.get("/documents")
def list_documents() -> list[dict]:
    with database() as db:
        rows = db.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        result = []
        for row in rows:
            counts = db.execute("SELECT COUNT(*) AS chunks FROM chunks WHERE document_id=?", (row["id"],)).fetchone()
            result.append({"id": row["id"], "name": row["name"], "status": row["status"],
                           "pages": row["page_count"], "chunks": counts["chunks"], "error": row["error"]})
        return result


@app.get("/documents/{document_id}")
def get_document(document_id: str) -> dict:
    with database() as db:
        row = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Document not found.")
        return {"id": row["id"], "name": row["name"], "status": row["status"], "pages": row["page_count"], "error": row["error"]}


@app.post("/documents/upload", status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF documents are accepted.")
    content = await file.read()
    if not content:
        raise HTTPException(400, "The selected document is empty.")
    document_id = str(uuid.uuid4())
    path = UPLOAD_DIR / f"{document_id}.pdf"
    try:
        with fitz.open(stream=content, filetype="pdf") as pdf:
            if pdf.page_count == 0:
                raise HTTPException(400, "The PDF contains no pages.")
    except fitz.FileDataError as exc:
        raise HTTPException(400, "The uploaded file is not a valid PDF.") from exc
    path.write_bytes(content)
    with database() as db:
        db.execute("INSERT INTO documents(id,name,path,status,created_at) VALUES(?,?,?,?,?)",
                   (document_id, Path(file.filename).name, str(path), "UPLOADING", utc_now()))
    background_tasks.add_task(extract_document, document_id)
    return {"id": document_id, "name": Path(file.filename).name, "status": "PROCESSING", "pages": 0, "error": None}


@app.post("/documents/{document_id}/process")
def process_document(document_id: str, background_tasks: BackgroundTasks) -> dict:
    with database() as db:
        row = db.execute("SELECT id,status FROM documents WHERE id=?", (document_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Document not found.")
        if row["status"] == "PROCESSING":
            return {"id": document_id, "status": "PROCESSING"}
        if row["status"] == "READY":
            raise HTTPException(409, "Document is already processed.")
        db.execute("UPDATE documents SET status='UPLOADING' WHERE id=?", (document_id,))
    background_tasks.add_task(extract_document, document_id)
    return {"id": document_id, "status": "PROCESSING"}


@app.get("/knowledge-graph")
def knowledge_graph(document_id: str | None = None) -> dict:
    with database() as db:
        if document_id:
            nodes = [
                dict(row)
                for row in db.execute(
                    "SELECT DISTINCT c.id,c.name,c.name AS label,c.description "
                    "FROM concepts c "
                    "JOIN concept_evidence ce ON ce.concept_id=c.id "
                    "JOIN chunks ch ON ch.id=ce.chunk_id "
                    "WHERE ch.document_id=? "
                    "ORDER BY c.name",
                    (document_id,),
                ).fetchall()
            ]

            edges = [
                dict(row)
                for row in db.execute(
                    "SELECT DISTINCT r.id,r.source_id,r.target_id,"
                    "r.relationship_type AS type,r.confidence,r.status,r.claim "
                    "FROM relationships r "
                    "JOIN relationship_evidence re ON re.relationship_id=r.id "
                    "JOIN chunks ch ON ch.id=re.chunk_id "
                    "WHERE ch.document_id=? "
                    "ORDER BY r.created_at",
                    (document_id,),
                ).fetchall()
            ]
        else:
            nodes = [
                dict(row)
                for row in db.execute(
                    "SELECT id,name,name AS label,description "
                    "FROM concepts ORDER BY name"
                ).fetchall()
            ]

            edges = [
                dict(row)
                for row in db.execute(
                    "SELECT r.id,r.source_id,r.target_id,"
                    "r.relationship_type AS type,r.confidence,r.status,r.claim "
                    "FROM relationships r ORDER BY r.created_at"
                ).fetchall()
            ]

        for edge in edges:
            source = db.execute(
                "SELECT name FROM concepts WHERE id=?", (edge["source_id"],)
            ).fetchone()
            target = db.execute(
                "SELECT name FROM concepts WHERE id=?", (edge["target_id"],)
            ).fetchone()

            edge["source"] = source["name"] if source else edge["source_id"]
            edge["target"] = target["name"] if target else edge["target_id"]

            edge.pop("source_id", None)
            edge.pop("target_id", None)

        return {"nodes": nodes, "edges": edges}


@app.get("/knowledge-gaps")
def knowledge_gaps() -> list[dict]:
    with database() as db:
        return [gap_row(row) for row in db.execute("SELECT * FROM gaps WHERE status='OPEN' ORDER BY created_at")]


@app.get("/evidence")
def list_evidence(document_id: str | None = None) -> list[dict]:
    with database() as db:
        if document_id:
            rows = db.execute(
                "SELECT DISTINCT c.id,c.page_number,c.text,d.name "
                "FROM chunks c "
                "JOIN documents d ON d.id=c.document_id "
                "WHERE d.status='READY' AND c.document_id=? "
                "ORDER BY c.page_number",
                (document_id,),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT DISTINCT c.id,c.page_number,c.text,d.name "
                "FROM chunks c "
                "JOIN documents d ON d.id=c.document_id "
                "WHERE d.status='READY' "
                "ORDER BY d.name,c.page_number"
            ).fetchall()

        return [
            {
                "id": row["id"],
                "source": row["name"],
                "document": row["name"],
                "page": row["page_number"],
                "text": row["text"],
                "relevance": "Source chunk",
                "type": "source",
                "contradiction": False,
            }
            for row in rows
        ]


@app.get("/retrieval/search")
def search_evidence(
    query: str,
    limit: int = 8,
    document_id: str | None = None,
) -> list[dict]:
    if not query.strip():
        raise HTTPException(400, "A non-empty search query is required.")

    if not 1 <= limit <= 30:
        raise HTTPException(400, "limit must be between 1 and 30.")

    return retrieve(
        query.strip(),
        limit,
        document_id=document_id,
    )

@app.get("/state")
def current_state() -> dict:
    with database() as db:
        investigations = [dict(row) for row in db.execute("SELECT * FROM investigations ORDER BY created_at DESC LIMIT 20")]
        counts = {
            "documents": db.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"],
            "concepts": db.execute("SELECT COUNT(*) AS n FROM concepts").fetchone()["n"],
            "relationships": db.execute("SELECT COUNT(*) AS n FROM relationships").fetchone()["n"],
            "gaps": db.execute("SELECT COUNT(*) AS n FROM gaps WHERE status='OPEN'").fetchone()["n"],
            "claims": db.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"],
        }
    return {"documents": list_documents(), "graph": knowledge_graph(), "gaps": knowledge_gaps(),
            "evidence": list_evidence(), "investigations": investigations, "counts": counts}


@app.post("/investigations/start", status_code=202)
def start_investigation(
    gap_id: str | None = None,
    document_id: str | None = None,
):
    with database() as db:
        gap = db.execute(
            "SELECT * FROM gaps "
            "WHERE status='OPEN' "
            "AND (? IS NULL OR id=?) "
            "AND (? IS NULL OR document_id=?) "
            "ORDER BY created_at LIMIT 1",
            (gap_id, gap_id, document_id, document_id),
        ).fetchone()

        if not gap:
            raise HTTPException(
                409,
                "No open evidence-grounded knowledge gaps are available "
                "for the selected document."
            )

        if document_id:
            document = db.execute(
                "SELECT id FROM documents "
                "WHERE id=? AND status='READY'",
                (document_id,),
            ).fetchone()

            if not document:
                raise HTTPException(
                    404,
                    "Selected document was not found or is not ready."
                )
        elif not db.execute(
            "SELECT 1 FROM documents WHERE status='READY' LIMIT 1"
        ).fetchone():
            raise HTTPException(
                409,
                "Process at least one PDF before starting an investigation."
            )

        require_llm()

        investigation_id = str(uuid.uuid4())

        db.execute(
            "INSERT INTO investigations("
            "id,gap_id,document_id,status,stage,created_at"
            ") VALUES(?,?,?,?,?,?)",
            (
                investigation_id,
                gap["id"],
                document_id or gap["document_id"],
                "RUNNING",
                "GAP",
                utc_now(),
            ),
        )

        return {
            "id": investigation_id,
            "gap_id": gap["id"],
            "document_id": document_id or gap["document_id"],
            "gap": gap["description"],
            "status": "RUNNING",
            "stage": "GAP",
        }


@app.post("/investigations/{investigation_id}/run", status_code=202)
def run_investigation_endpoint(investigation_id: str, background_tasks: BackgroundTasks) -> dict:
    with database() as db:
        row = db.execute("SELECT id,status FROM investigations WHERE id=?", (investigation_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Investigation not found.")
        if row["status"] != "RUNNING":
            raise HTTPException(409, "Investigation is not in a runnable state.")
    background_tasks.add_task(run_investigation, investigation_id)
    return {"id": investigation_id, "status": "RUNNING"}


@app.get("/investigations")
def list_investigations() -> list[dict]:
    with database() as db:
        return [dict(row) for row in db.execute("SELECT * FROM investigations ORDER BY created_at DESC")]


@app.get("/investigations/{investigation_id}")
def get_investigation(investigation_id: str) -> dict:
    with database() as db:
        row = db.execute("SELECT i.*,g.description AS gap FROM investigations i LEFT JOIN gaps g ON g.id=i.gap_id WHERE i.id=?", (investigation_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Investigation not found.")
        return dict(row)


@app.get("/investigations/{investigation_id}/evidence")
def get_investigation_evidence(investigation_id: str) -> list[dict]:
    with database() as db:
        rows = db.execute(
            "SELECT ie.evidence_type,ie.relevance,c.id,c.page_number,c.text,d.name FROM investigation_evidence ie "
            "JOIN chunks c ON c.id=ie.chunk_id JOIN documents d ON d.id=c.document_id WHERE ie.investigation_id=? "
            "ORDER BY ie.evidence_type,c.page_number", (investigation_id,)
        ).fetchall()
        return [{"id": row["id"], "source": row["name"], "document": row["name"], "page": row["page_number"],
                 "text": row["text"], "relevance": f"{row['relevance']:.2f}", "type": row["evidence_type"],
                 "contradiction": row["evidence_type"] == "counter"} for row in rows]


@app.post("/knowledge-graph/update")
def update_graph_endpoint() -> dict:
    return {"gaps": detect_gaps(), "graph": knowledge_graph()}


@app.get("/investigations/{investigation_id}/verification")
def get_verification(investigation_id: str) -> dict:
    investigation = get_investigation(investigation_id)
    evidence = get_investigation_evidence(investigation_id)
    return {"status": investigation["status"], "confidence": investigation["confidence"],
            "supporting_evidence": [item for item in evidence if item["type"] == "supporting"],
            "counter_evidence": [item for item in evidence if item["type"] == "counter"],
            "reasoning_summary": investigation["reasoning_summary"]}


@app.exception_handler(ValidationError)
async def validation_error_handler(_, exc: ValidationError):
    return JSONResponse(status_code=422, content={"detail": f"Structured extraction validation failed: {exc}"})