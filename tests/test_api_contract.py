from fastapi.testclient import TestClient

import api_server


client = TestClient(api_server.app)


def test_whitespace_question_is_rejected_without_rag(monkeypatch):
    monkeypatch.setattr(
        api_server,
        "traced_chat",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("RAG must not run")),
    )
    response = client.post("/chat", json={"question": "   "})
    assert response.status_code == 422
    assert response.json()["detail"] == "问题不能为空。"


def test_overlong_question_is_rejected_by_schema():
    response = client.post("/chat", json={"question": "x" * 8001})
    assert response.status_code == 422


def test_direct_llm_contract_marks_citations_unsupported(monkeypatch):
    monkeypatch.setattr(api_server, "chat_messages", lambda *args, **kwargs: "direct answer")
    response = client.post("/llm/chat", json={"prompt": "hello"})
    assert response.status_code == 200
    assert response.json() == {
        "model": api_server.settings.llm_model,
        "answer": "direct answer",
        "mode": "direct_llm",
        "citations_supported": False,
    }
