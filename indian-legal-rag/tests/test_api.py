from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def record():
    return {
        "record_id": "a" * 64,
        "law_name": "constitution",
        "source_file": "constitution_qa.json",
        "question": "What is the fixture value?",
        "answer": "The fixture value is blue.",
        "section_number": None,
        "article_number": None,
        "law_version": None,
        "effective_date": None,
        "source_url": None,
        "chapter": None,
        "source_status": "dataset_unverified",
        "retrieval_method": "semantic",
        "rank": 1,
        "score": 0.8,
    }


class StubRetriever:
    def __init__(self, result):
        self.result = result

    def retrieve(self, *args, **kwargs):
        return self.result


class StubLLM:
    async def generate(self, messages):
        return f"Grounded answer [{record()['record_id']}]"


class InvalidCitationLLM:
    async def generate(self, messages):
        return f"Grounded answer [{'f' * 64}]"


def app_for(result, *, llm_configured=False):
    settings = Settings(
        _env_file=None,
        llm_base_url="https://example.invalid/v1" if llm_configured else None,
        llm_model="test-model" if llm_configured else None,
    )
    application = create_app(settings)
    application.state.services._retriever = StubRetriever(result)
    application.state.services.llm = StubLLM()
    return application


def result(evidence, insufficient=False):
    return {
        "evidence": evidence,
        "retrieval": {"method": "semantic", "evidence_count": len(evidence)},
        "insufficient_evidence": insufficient,
    }


def test_health_and_qdrant_missing_configuration():
    client = TestClient(create_app(Settings(_env_file=None)))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/health/qdrant").status_code == 503


def test_retrieve_returns_sources_without_calling_llm():
    client = TestClient(app_for(result([record()])))
    response = client.post("/retrieve", json={"question": "What is this?"})
    assert response.status_code == 200
    assert response.json()["evidence"][0]["record_id"] == record()["record_id"]


def test_api_rejects_blank_or_unreasonably_long_questions():
    client = TestClient(app_for(result([])))
    assert client.post("/retrieve", json={"question": ""}).status_code == 422
    assert client.post("/retrieve", json={"question": "x" * 4001}).status_code == 422


def test_ask_requires_llm_configuration():
    client = TestClient(app_for(result([record()])))
    response = client.post("/ask", json={"question": "Question"})
    assert response.status_code == 503
    assert "LLM_BASE_URL" in response.json()["detail"]


def test_ask_abstains_without_evidence():
    client = TestClient(app_for(result([], insufficient=True), llm_configured=True))
    response = client.post("/ask", json={"question": "Question"})
    assert response.status_code == 200
    assert response.json()["insufficient_evidence"] is True
    assert response.json()["sources"] == []


def test_ask_returns_citation_from_retrieved_record():
    client = TestClient(app_for(result([record()]), llm_configured=True))
    response = client.post("/ask", json={"question": "Question"})
    assert response.status_code == 200
    body = response.json()
    assert body["sources"][0]["record_id"] == record()["record_id"]
    assert body["insufficient_evidence"] is False


def test_ask_rejects_citation_not_in_retrieved_evidence():
    application = app_for(result([record()]), llm_configured=True)
    application.state.services.llm = InvalidCitationLLM()
    response = TestClient(application).post("/ask", json={"question": "Question"})
    assert response.status_code == 502
    assert "source validation" in response.json()["detail"]
