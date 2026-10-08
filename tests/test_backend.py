import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pymupdf as fitz
from fastapi.testclient import TestClient

import backend


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.original_db_path = backend.DB_PATH
        self.original_upload_dir = backend.UPLOAD_DIR
        backend.DB_PATH = Path(self.temporary_directory.name) / "test.sqlite3"
        backend.UPLOAD_DIR = Path(self.temporary_directory.name) / "uploads"
        backend.UPLOAD_DIR.mkdir()
        backend.initialize_database()
        self.saved_env = {key: os.environ.get(key) for key in (
            "GEMINI_API_KEY", "GEMINI_MODEL", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
            "EMBEDDING_MODEL", "EMBEDDING_PROVIDER",
        )}
        for key in self.saved_env:
            os.environ.pop(key, None)
        self.client = TestClient(backend.app)

    def tearDown(self):
        self.client.close()
        backend.DB_PATH = self.original_db_path
        backend.UPLOAD_DIR = self.original_upload_dir
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temporary_directory.cleanup()

    def create_pdf(self, path: Path, text: str) -> None:
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), text)
        pdf.save(path)
        pdf.close()

    def test_chunking_preserves_text_and_limit(self):
        text = "One meaningful paragraph.\n\n" + "battery " * 400
        chunks = backend.split_chunks(text, limit=100)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertIn("One meaningful paragraph.", " ".join(chunks))

    def test_cosine_similarity(self):
        self.assertAlmostEqual(backend.cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(backend.cosine([1, 0], [0, 1]), 0.0)

    def test_retrieval_ranks_persisted_vectors_by_cosine_similarity(self):
        document_id = str(uuid.uuid4())
        first_chunk = str(uuid.uuid4())
        second_chunk = str(uuid.uuid4())
        with backend.database() as db:
            db.execute("INSERT INTO documents(id,name,path,status,created_at) VALUES(?,?,?,?,?)",
                       (document_id, "retrieval.pdf", "unused.pdf", "READY", backend.utc_now()))
            db.executemany(
                "INSERT INTO chunks(id,document_id,page_number,text,embedding) VALUES(?,?,?,?,?)",
                [(first_chunk, document_id, 1, "closest semantic evidence", json.dumps([1.0, 0.0])),
                 (second_chunk, document_id, 2, "different evidence", json.dumps([0.0, 1.0]))],
            )

        with patch.object(backend, "embed_texts", return_value=[[1.0, 0.0]]):
            results = backend.retrieve("query", limit=2)

        self.assertEqual(results[0]["id"], first_chunk)
        self.assertEqual(results[0]["page"], 1)
        self.assertEqual(results[0]["document"], "retrieval.pdf")

    def test_local_embedding_generation_uses_sentence_transformer_model(self):
        class LocalModel:
            def __init__(self):
                self.options = None

            def encode(self, texts, **options):
                self.options = (texts, options)
                return [[1 / (384 ** 0.5)] * 384 for _ in texts]

        model = LocalModel()
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "local"}), \
             patch.object(backend, "local_embedding_model", return_value=model):
            vectors = backend.embed_texts(["source chunk"])

        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 384)
        self.assertEqual(model.options[0], ["source chunk"])
        self.assertTrue(model.options[1]["normalize_embeddings"])

    def create_document_record(self, name: str, text: str) -> str:
        path = backend.UPLOAD_DIR / name
        self.create_pdf(path, text)
        document_id = str(uuid.uuid4())
        with backend.database() as db:
            db.execute("INSERT INTO documents(id,name,path,status,created_at) VALUES(?,?,?,?,?)",
                       (document_id, name, str(path), "UPLOADING", backend.utc_now()))
        return document_id

    def create_failed_document_with_vectors(self, name: str, chunk_count: int = 1) -> tuple[str, list[str], list[list[float]]]:
        document_id = self.create_document_record(name, "Persisted source text for Gemini retry.")
        chunk_ids = [str(uuid.uuid4()) for _ in range(chunk_count)]
        vectors = [[0.1, 0.2, 0.3] for _ in range(chunk_count)]
        with backend.database() as db:
            db.executemany(
                "INSERT INTO chunks(id,document_id,page_number,text,embedding) VALUES(?,?,?,?,?)",
                [(chunk_id, document_id, index + 1, f"Persisted source chunk {index + 1}", json.dumps(vector))
                 for index, (chunk_id, vector) in enumerate(zip(chunk_ids, vectors))],
            )
            db.execute("UPDATE documents SET status='FAILED',page_count=? WHERE id=?", (chunk_count, document_id))
        return document_id, chunk_ids, vectors

    @staticmethod
    def empty_extraction(_client, _system, _user, **_kwargs):
        return {"concepts": [], "relationships": [], "claims": [], "contradictions": []}

    def test_chunks_are_persisted_before_embedding_is_called(self):
        document_id = self.create_document_record("ordered.pdf", "Actual source chunk before vector generation.")
        observed = {}

        def embed(texts):
            with backend.database() as db:
                rows = db.execute("SELECT page_number,text,embedding FROM chunks WHERE document_id=?", (document_id,)).fetchall()
            observed["rows"] = [dict(row) for row in rows]
            return [[0.1, 0.2, 0.3] for _ in texts]

        with patch.object(backend, "embed_texts", side_effect=embed), \
             patch.object(backend, "require_llm", return_value=object()), \
             patch.object(backend, "model_json", side_effect=self.empty_extraction):
            backend.extract_document(document_id)

        self.assertEqual(len(observed["rows"]), 1)
        self.assertEqual(observed["rows"][0]["page_number"], 1)
        self.assertIn("Actual source chunk", observed["rows"][0]["text"])
        self.assertEqual(observed["rows"][0]["embedding"], "[]")

    def test_embedding_failure_keeps_chunks_for_retry(self):
        document_id = self.create_document_record("embedding-failure.pdf", "Persist this source chunk first.")

        def fail_embedding(_texts):
            with backend.database() as db:
                count = db.execute("SELECT COUNT(*) AS n FROM chunks WHERE document_id=?", (document_id,)).fetchone()["n"]
            self.assertEqual(count, 1)
            raise RuntimeError("local model unavailable")

        with patch.object(backend, "embed_texts", side_effect=fail_embedding):
            backend.extract_document(document_id)

        with backend.database() as db:
            document = db.execute("SELECT status,error FROM documents WHERE id=?", (document_id,)).fetchone()
            chunks = db.execute("SELECT COUNT(*) AS n FROM chunks WHERE document_id=?", (document_id,)).fetchone()["n"]
            vector = db.execute("SELECT embedding FROM chunks WHERE document_id=?", (document_id,)).fetchone()["embedding"]
        self.assertEqual(document["status"], "FAILED")
        self.assertIn("local model unavailable", document["error"])
        self.assertEqual(chunks, 1)
        self.assertEqual(vector, "[]")

    def test_retrying_failed_document_reuses_chunks_and_saves_embeddings(self):
        document_id = self.create_document_record("retry.pdf", "Reprocess the existing unembedded source chunk.")
        with patch.object(backend, "embed_texts", side_effect=RuntimeError("temporary embedding failure")):
            backend.extract_document(document_id)

        def retry_embed(texts):
            with backend.database() as db:
                count = db.execute("SELECT COUNT(*) AS n FROM chunks WHERE document_id=?", (document_id,)).fetchone()["n"]
            self.assertEqual(count, 1)
            return [[0.4, 0.5, 0.6] for _ in texts]

        with patch.object(backend, "embed_texts", side_effect=retry_embed), \
             patch.object(backend, "require_llm", return_value=object()), \
             patch.object(backend, "model_json", side_effect=self.empty_extraction):
            response = self.client.post(f"/documents/{document_id}/process")

        self.assertEqual(response.status_code, 200)
        with backend.database() as db:
            document = db.execute("SELECT status FROM documents WHERE id=?", (document_id,)).fetchone()
            chunks = db.execute("SELECT COUNT(*) AS n FROM chunks WHERE document_id=?", (document_id,)).fetchone()["n"]
            vector = json.loads(db.execute("SELECT embedding FROM chunks WHERE document_id=?", (document_id,)).fetchone()["embedding"])
        self.assertEqual(document["status"], "READY")
        self.assertEqual(chunks, 1)
        self.assertEqual(vector, [0.4, 0.5, 0.6])

    def test_gemini_structured_extraction_uses_schema_and_preserves_vectors(self):
        document_id, chunk_ids, vectors = self.create_failed_document_with_vectors("gemini-success.pdf", 2)
        extraction = {
            "concepts": [
                {"name": "Battery Safety", "description": "Safety evidence", "evidence_chunk_ids": [chunk_ids[0]]},
                {"name": "Thermal Runaway", "description": "Thermal evidence", "evidence_chunk_ids": [chunk_ids[1]]},
            ],
            "relationships": [{"source": "Battery Safety", "target": "Thermal Runaway", "type": "mitigates",
                               "confidence": 0.9, "claim": "Safety controls reduce risk.", "evidence_chunk_ids": [chunk_ids[0]]}],
            "claims": [{"text": "Controls reduce risk.", "evidence_chunk_id": chunk_ids[0]}],
            "contradictions": [],
        }
        generate_content = Mock(return_value=SimpleNamespace(text=json.dumps(extraction)))
        fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        embedding_mock = Mock(side_effect=AssertionError("stored vectors must be reused"))

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-2.5-flash"}), \
             patch.object(backend, "require_llm", return_value=fake_client), \
             patch.object(backend, "embed_texts", embedding_mock):
            backend.extract_document(document_id)

        config = generate_content.call_args.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertIs(config.response_schema, backend.Extraction)
        self.assertEqual(generate_content.call_args.kwargs["model"], "gemini-2.5-flash")
        with backend.database() as db:
            document = db.execute("SELECT status FROM documents WHERE id=?", (document_id,)).fetchone()
            concepts = db.execute("SELECT name FROM concepts ORDER BY name").fetchall()
            relationships = db.execute("SELECT COUNT(*) AS n FROM relationships").fetchone()["n"]
            claims = db.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"]
            saved_vectors = [json.loads(row["embedding"]) for row in db.execute(
                "SELECT embedding FROM chunks WHERE document_id=? ORDER BY page_number", (document_id,)
            )]
        self.assertEqual(document["status"], "READY")
        self.assertEqual([row["name"] for row in concepts], ["Battery Safety", "Thermal Runaway"])
        self.assertEqual(relationships, 1)
        self.assertEqual(claims, 1)
        self.assertEqual(saved_vectors, vectors)
        embedding_mock.assert_not_called()

    def test_malformed_gemini_response_writes_no_partial_graph_and_retry_reuses_vectors(self):
        document_id, chunk_ids, vectors = self.create_failed_document_with_vectors("gemini-malformed.pdf", 6)
        first_group_id = chunk_ids[0]
        partial_extraction = {
            "concepts": [{"name": "Uncommitted concept", "description": "Must not persist", "evidence_chunk_ids": [first_group_id]}],
            "relationships": [], "claims": [], "contradictions": [],
        }
        generate_content = Mock(side_effect=[
            SimpleNamespace(text=json.dumps(partial_extraction)),
            SimpleNamespace(text="{malformed JSON"),
        ])
        fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        embedding_mock = Mock(side_effect=AssertionError("stored vectors must be reused"))
        environment = {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-2.5-flash"}

        with patch.dict(os.environ, environment), \
             patch.object(backend, "require_llm", return_value=fake_client), \
             patch.object(backend, "embed_texts", embedding_mock):
            backend.extract_document(document_id)
            with backend.database() as db:
                failed = db.execute("SELECT status,error FROM documents WHERE id=?", (document_id,)).fetchone()
                graph_counts = [db.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                                for table in ("concepts", "relationships", "claims")]
                saved_vectors = [json.loads(row["embedding"]) for row in db.execute(
                    "SELECT embedding FROM chunks WHERE document_id=? ORDER BY page_number", (document_id,)
                )]
            self.assertEqual(failed["status"], "FAILED")
            self.assertIn("Gemini returned malformed JSON", failed["error"])
            self.assertEqual(graph_counts, [0, 0, 0])
            self.assertEqual(saved_vectors, vectors)

            retry_calls = 0

            def successful_retry(**kwargs):
                nonlocal retry_calls
                retry_calls += 1
                group = json.loads(kwargs["contents"])
                result = {
                    "concepts": [{"name": f"Retry concept {retry_calls}", "description": "Validated source concept",
                                  "evidence_chunk_ids": [group[0]["chunk_id"]]}],
                    "relationships": [], "claims": [], "contradictions": [],
                }
                return SimpleNamespace(text=json.dumps(result))

            generate_content.side_effect = successful_retry
            response = self.client.post(f"/documents/{document_id}/process")

        self.assertEqual(response.status_code, 200)
        with backend.database() as db:
            document = db.execute("SELECT status FROM documents WHERE id=?", (document_id,)).fetchone()
            concepts = db.execute("SELECT name FROM concepts ORDER BY name").fetchall()
        self.assertEqual(document["status"], "READY")
        self.assertEqual([row["name"] for row in concepts], ["Retry concept 1", "Retry concept 2"])
        embedding_mock.assert_not_called()

    def test_gemini_quota_failure_is_retryable_without_embedding_calls(self):
        document_id, chunk_ids, vectors = self.create_failed_document_with_vectors("gemini-quota.pdf")

        class QuotaError(Exception):
            code = 429

        generate_content = Mock(side_effect=QuotaError("RESOURCE_EXHAUSTED: quota exceeded"))
        fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        embedding_mock = Mock(side_effect=AssertionError("stored vectors must be reused"))
        environment = {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-2.5-flash"}

        with patch.dict(os.environ, environment), \
             patch.object(backend, "require_llm", return_value=fake_client), \
             patch.object(backend, "embed_texts", embedding_mock):
            backend.extract_document(document_id)
            with backend.database() as db:
                failed = db.execute("SELECT status,error FROM documents WHERE id=?", (document_id,)).fetchone()
                graph_counts = [db.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                                for table in ("concepts", "relationships")]
                saved_vector = json.loads(db.execute("SELECT embedding FROM chunks WHERE id=?", (chunk_ids[0],)).fetchone()["embedding"])
            self.assertEqual(failed["status"], "FAILED")
            self.assertIn("Gemini quota or rate limit reached", failed["error"])
            self.assertEqual(graph_counts, [0, 0])
            self.assertEqual(saved_vector, vectors[0])

            generate_content.side_effect = None
            generate_content.return_value = SimpleNamespace(text=json.dumps({
                "concepts": [{"name": "Quota retry concept", "description": "Recovered after quota reset",
                              "evidence_chunk_ids": [chunk_ids[0]]}],
                "relationships": [], "claims": [], "contradictions": [],
            }))
            response = self.client.post(f"/documents/{document_id}/process")

        self.assertEqual(response.status_code, 200)
        with backend.database() as db:
            document = db.execute("SELECT status FROM documents WHERE id=?", (document_id,)).fetchone()
            concept_count = db.execute("SELECT COUNT(*) AS n FROM concepts").fetchone()["n"]
        self.assertEqual(document["status"], "READY")
        self.assertEqual(concept_count, 1)
        embedding_mock.assert_not_called()

    def test_live_state_starts_empty(self):
        response = self.client.get("/state")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["counts"], {
            "documents": 0, "concepts": 0, "relationships": 0, "gaps": 0, "claims": 0
        })

    def test_invalid_pdf_is_rejected(self):
        response = self.client.post("/documents/upload", files={"file": ("broken.pdf", b"not a PDF", "application/pdf")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("valid PDF", response.json()["detail"])

    def test_valid_pdf_fails_closed_without_api_key(self):
        path = backend.UPLOAD_DIR / "source.pdf"
        self.create_pdf(path, "Battery research source text.")
        document_id = str(uuid.uuid4())
        with backend.database() as db:
            db.execute("INSERT INTO documents(id,name,path,status,created_at) VALUES(?,?,?,?,?)",
                       (document_id, "source.pdf", str(path), "UPLOADING", backend.utc_now()))

        with patch.object(backend, "embed_texts", return_value=[[1.0, 0.0]]):
            backend.extract_document(document_id)
        with backend.database() as db:
            document = db.execute("SELECT status,page_count,error FROM documents WHERE id=?", (document_id,)).fetchone()
            chunks = db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            concepts = db.execute("SELECT COUNT(*) AS n FROM concepts").fetchone()["n"]
        self.assertEqual(document["status"], "FAILED")
        self.assertEqual(document["page_count"], 1)
        self.assertIn("GEMINI_API_KEY is missing", document["error"])
        self.assertEqual(chunks, 1)
        self.assertEqual(concepts, 0)

    def test_empty_pdf_is_rejected_before_model_configuration(self):
        path = backend.UPLOAD_DIR / "empty.pdf"
        pdf = fitz.open()
        pdf.new_page()
        pdf.save(path)
        pdf.close()
        document_id = str(uuid.uuid4())
        with backend.database() as db:
            db.execute("INSERT INTO documents(id,name,path,status,created_at) VALUES(?,?,?,?,?)",
                       (document_id, "empty.pdf", str(path), "UPLOADING", backend.utc_now()))

        backend.extract_document(document_id)
        with backend.database() as db:
            document = db.execute("SELECT status,error FROM documents WHERE id=?", (document_id,)).fetchone()
        self.assertEqual(document["status"], "FAILED")
        self.assertIn("no extractable text", document["error"])

    def test_ingestion_persists_chunk_provenance_and_detects_a_corpus_gap(self):
        path = backend.UPLOAD_DIR / "research.pdf"
        self.create_pdf(path, "Battery Degradation and Thermal Runaway were observed in the same study.")
        document_id = str(uuid.uuid4())
        with backend.database() as db:
            db.execute("INSERT INTO documents(id,name,path,status,created_at) VALUES(?,?,?,?,?)",
                       (document_id, "research.pdf", str(path), "UPLOADING", backend.utc_now()))

        def extraction(_client, _system, user):
            supplied = json.loads(user)
            chunk_id = supplied[0]["chunk_id"]
            return {
                "concepts": [
                    {"name": "Battery Degradation", "description": "Battery aging", "evidence_chunk_ids": [chunk_id]},
                    {"name": "Thermal Runaway", "description": "Thermal event", "evidence_chunk_ids": [chunk_id]},
                ],
                "relationships": [], "claims": [], "contradictions": [],
            }

        with patch.object(backend, "require_llm", return_value=object()):
            with patch.object(backend, "embed_texts", return_value=[[1.0, 0.0]]):
                with patch.object(backend, "model_json", side_effect=extraction):
                    backend.extract_document(document_id)

        with backend.database() as db:
            document = db.execute("SELECT status,page_count FROM documents WHERE id=?", (document_id,)).fetchone()
            chunk = db.execute("SELECT id,page_number,text,embedding FROM chunks WHERE document_id=?", (document_id,)).fetchone()
            concept_evidence_count = db.execute("SELECT COUNT(*) AS n FROM concept_evidence").fetchone()["n"]
            gaps = db.execute("SELECT type,description FROM gaps WHERE status='OPEN'").fetchall()
        self.assertEqual(document["status"], "READY")
        self.assertEqual(document["page_count"], 1)
        self.assertEqual(chunk["page_number"], 1)
        self.assertIn("Battery Degradation", chunk["text"])
        self.assertEqual(chunk["embedding"], "[1.0, 0.0]")
        self.assertEqual(concept_evidence_count, 2)
        self.assertTrue(any(gap["type"] == "Missing Relationship" for gap in gaps))

    def test_upload_surfaces_missing_api_key(self):
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "Real source text.")
        content = pdf.tobytes()
        pdf.close()
        with patch.object(backend, "embed_texts", return_value=[[1.0, 0.0]]):
            response = self.client.post("/documents/upload", files={"file": ("source.pdf", content, "application/pdf")})
        self.assertEqual(response.status_code, 202)
        listed = self.client.get("/documents").json()
        self.assertEqual(listed[0]["status"], "FAILED")
        self.assertIn("GEMINI_API_KEY is missing", listed[0]["error"])


if __name__ == "__main__":
    unittest.main()