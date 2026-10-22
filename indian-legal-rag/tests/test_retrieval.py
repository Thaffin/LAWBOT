from types import SimpleNamespace

from app.chunking.qa_chunker import chunk_record
from app.config import Settings
from app.retrieval.exact_search import ExactSearchIndex
from app.retrieval.hybrid_search import reciprocal_rank_fusion
from app.retrieval.retriever import Retriever
from app.retrieval.semantic_search import SemanticSearch
from app.vectordb.vector_store import QdrantVectorStore


def record():
    return {
        "record_id": "a" * 64,
        "question": "What is the fixture value?",
        "answer": "The fixture value is blue.",
        "source_file": "constitution_qa.json",
        "law_name": "constitution",
        "record_type": "qa_pair",
        "article_number": None,
        "section_number": None,
        "chapter": None,
        "law_version": None,
        "effective_date": None,
        "source_url": None,
        "source_status": "dataset_unverified",
        "content_hash": "b" * 64,
    }


def test_qa_chunk_keeps_question_answer_and_provenance_together():
    item = record()
    chunks = chunk_record(item, Settings(_env_file=None))
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Question: What is the fixture value?\nAnswer: The fixture value is blue."
    assert chunks[0]["payload"]["question"] == item["question"]
    assert chunks[0]["payload"]["answer"] == item["answer"]
    assert chunks[0]["payload"]["parent_record_id"] == item["record_id"]


def test_long_qa_splits_on_word_boundaries_and_retains_original_answer():
    item = record()
    item["answer"] = " ".join(["fixtureword"] * 60)
    settings = Settings(_env_file=None, chunk_size_chars=256, chunk_overlap_chars=8)
    chunks = chunk_record(item, settings)
    assert len(chunks) > 1
    assert all(chunk["payload"]["answer"] == item["answer"] for chunk in chunks)
    assert all(chunk["payload"]["parent_record_id"] == item["record_id"] for chunk in chunks)


def test_exact_question_and_article_lookup(tmp_path):
    index = ExactSearchIndex(tmp_path / "exact.sqlite3")
    item = record()
    item["article_number"] = "21"
    index.sync([item])
    exact = index.search("  WHAT is the fixture value? ")
    article = index.search("Explain Article 21", law_name="constitution")
    assert exact[0]["record_id"] == item["record_id"]
    assert exact[0]["retrieval_method"] == "exact_question"
    assert article[0]["retrieval_method"] == "explicit_article"
    assert index.search("Article 21", law_name="ipc") == []


def test_rrf_fuses_ranks_and_deduplicates_records():
    exact = [{"record_id": "r1", "retrieval_method": "exact_question", "rank": 1}]
    semantic = [
        {"record_id": "r1", "retrieval_method": "semantic", "rank": 1, "score": 0.8},
        {"record_id": "r2", "retrieval_method": "semantic", "rank": 2, "score": 0.7},
    ]
    merged = reciprocal_rank_fusion([exact, semantic])
    assert [item["record_id"] for item in merged] == ["r1", "r2"]
    assert merged[0]["retrieval_method"] == "hybrid"
    assert merged[0]["exact_match"] is True
    assert merged[0]["rrf_score"] > merged[1]["rrf_score"]

def test_rrf_counts_one_parent_record_once_per_retrieval_method():
    results = [
        {"record_id": "r1", "retrieval_method": "semantic", "score": 0.9},
        {"record_id": "r1", "retrieval_method": "semantic", "score": 0.8},
    ]
    merged = reciprocal_rank_fusion([results])
    assert len(merged) == 1
    assert merged[0]["rrf_score"] == 1 / 61


def test_semantic_search_uses_query_embedding_and_filters():
    class Embedder:
        def encode_query(self, question):
            return [1.0, 0.0]

    class Store:
        def search(self, vector, limit, law_name, source_file):
            assert vector == [1.0, 0.0]
            assert (limit, law_name, source_file) == (4, "ipc", "ipc_qa.json")
            return [{"record_id": "r", "score": 0.9}]

    result = SemanticSearch(Embedder(), Store()).search(
        "Question", 4, "ipc", "ipc_qa.json"
    )
    assert result[0]["score"] == 0.9


def test_qdrant_semantic_search_returns_real_payload_with_filters():
    class MockQdrant:
        def query_points(self, **kwargs):
            assert kwargs["query"] == [0.25, 0.75]
            assert kwargs["limit"] == 3
            assert kwargs["query_filter"] is not None
            return SimpleNamespace(points=[
                SimpleNamespace(
                    id="point-1",
                    score=0.82,
                    payload={
                        "record_id": "r1",
                        "law_name": "ipc",
                        "source_file": "ipc_qa.json",
                        "question": "Question",
                        "answer": "Answer",
                        "source_status": "legacy_dataset_unverified",
                    },
                )
            ])

    store = QdrantVectorStore(MockQdrant(), Settings(_env_file=None))
    result = store.search([0.25, 0.75], 3, law_name="ipc")
    assert result[0]["record_id"] == "r1"
    assert result[0]["source_file"] == "ipc_qa.json"
    assert result[0]["retrieval_method"] == "semantic"
    assert result[0]["score"] == 0.82


def test_retriever_marks_empty_evidence_insufficient(tmp_path):
    class NoResults:
        def search(self, *args, **kwargs):
            return []

    dummy_embedder = type("Embedder", (), {"encode_query": lambda *_: [0.0]})()
    retriever = Retriever(
        Settings(_env_file=None),
        ExactSearchIndex(tmp_path / "exact.sqlite3"),
        SemanticSearch(dummy_embedder, NoResults()),
    )
    result = retriever.retrieve("No matching fixture question")
    assert result["evidence"] == []
    assert result["insufficient_evidence"] is True


def test_semantic_evidence_abstains_until_threshold_is_calibrated(tmp_path):
    class Semantic:
        def search(self, *args, **kwargs):
            return [{
                **record(),
                "score": 0.9,
                "retrieval_method": "semantic",
                "rank": 1,
            }]

    retriever = Retriever(
        Settings(_env_file=None, similarity_threshold=None),
        ExactSearchIndex(tmp_path / "exact.sqlite3"),
        SemanticSearch(type("Embedder", (), {"encode_query": lambda *_: [0.0]})(), Semantic()),
    )
    result = retriever.retrieve("Different question")
    assert result["evidence"]
    assert result["insufficient_evidence"] is True

    calibrated = Retriever(
        Settings(_env_file=None, similarity_threshold=0.8),
        ExactSearchIndex(tmp_path / "exact.sqlite3"),
        SemanticSearch(type("Embedder", (), {"encode_query": lambda *_: [0.0]})(), Semantic()),
    )
    assert calibrated.retrieve("Different question")["insufficient_evidence"] is False
