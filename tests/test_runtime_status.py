import json

from src.config import settings
from src.status import get_runtime_status


def test_status_is_redacted_and_reports_missing_index(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "one.pdf").write_bytes(b"%PDF-test")
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()

    monkeypatch.setattr(settings, "pdf_dir", str(pdf_dir))
    monkeypatch.setattr(settings, "chroma_dir", str(chroma_dir))
    monkeypatch.setattr(settings, "openai_api_key", "super-secret-value")
    monkeypatch.setattr(settings, "llm_model", "test-model")

    status = get_runtime_status(force=True)
    serialized = json.dumps(status)

    assert status["service"] == "literature-rag-api"
    assert status["knowledge_base"]["document_count"] == 1
    assert status["vector_index"]["status"] == "missing"
    assert status["llm"]["configured"] is True
    assert "super-secret-value" not in serialized
    assert "openai_api_key" not in serialized.lower()
