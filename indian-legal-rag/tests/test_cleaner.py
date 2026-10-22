import json

from app.ingestion.cleaner import (
    clean_dataset,
    clean_records,
    normalize_text,
    validate_sources,
)

def test_normalizes_whitespace_without_flattening_paragraphs():
    assert normalize_text("  first \t line \r\n\r\n second   line  ") == (
        "first line\n\nsecond line"
    )


def test_exact_duplicates_removed_and_incomplete_records_reported(caplog):
    rows = [
        {"question": " What is   this? ", "answer": " It is a test. "},
        {"question": "What is this?", "answer": "It is a test."},
        {"question": "", "answer": "Missing question."},
        {"question": "Missing answer?", "answer": ""},
    ]
    records, report = clean_records(rows, "constitution_qa.json")
    assert len(records) == 1
    assert report["duplicate_count"] == 1
    assert report["rejected_count"] == 2
    assert records[0]["question"] == "What is this?"
    assert records[0]["answer"] == "It is a test."
    assert "skipped" in caplog.text


def test_stable_ids_and_explicit_metadata_only():
    row = {"question": "What is the number 21?", "answer": "The number is 21."}
    first, _ = clean_records([row], "ipc_qa.json")
    second, _ = clean_records([row], "ipc_qa.json")
    assert first[0]["record_id"] == second[0]["record_id"]
    assert first[0]["article_number"] is None
    assert first[0]["section_number"] is None
    assert first[0]["source_status"] == "legacy_dataset_unverified"
    assert clean_records([row], "ipc_qa.json")[1]["missing_metadata_count"] == 1


def test_explicit_article_and_version_are_preserved():
    row = {
        "question": "What is protected?",
        "answer": "Article 21 protects life and liberty.",
        "id": "source-17",
        "article_number": "21",
        "law_version": "Provided by dataset",
        "effective_date": "Provided date",
        "source_url": "https://example.invalid/source",
    }
    records, _ = clean_records([row], "constitution_qa.json")
    assert records[0]["article_number"] == "21"
    assert records[0]["source_record_id"] == "source-17"
    assert records[0]["law_version"] == "Provided by dataset"
    assert records[0]["effective_date"] == "Provided date"
    assert records[0]["source_url"] == "https://example.invalid/source"


def test_clean_dataset_does_not_modify_input_and_writes_report(tmp_path):
    raw, cleaned, reports = tmp_path / "raw", tmp_path / "cleaned", tmp_path / "reports"
    raw.mkdir()
    original = '[{"question":"Q","answer":"A"}]'
    (raw / "constitution_qa.json").write_text(original, encoding="utf-8")
    for name in ("ipc_qa.json", "crpc_qa.json"):
        (raw / name).write_text("[]", encoding="utf-8")
    report = clean_dataset(raw, cleaned, reports)
    assert report["accepted_count"] == 1
    assert (raw / "constitution_qa.json").read_text(encoding="utf-8") == original
    assert json.loads((cleaned / "constitution_qa.json").read_text())[0]["law_name"] == "constitution"
    assert (reports / "cleaning_report.json").exists()


def test_validation_returns_false_for_missing_file(tmp_path):
    assert validate_sources(tmp_path)["valid"] is False
