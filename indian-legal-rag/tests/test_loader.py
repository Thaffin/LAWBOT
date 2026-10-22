from pathlib import Path

import pytest

from app.ingestion.loader import SourceDataError, load_json_file


def test_loads_synthetic_fixture():
    fixture = Path(__file__).parent / "fixtures" / "valid_qa.json"
    assert len(load_json_file(fixture)) == 2


def test_loads_array_of_qa_objects(tmp_path):
    path = tmp_path / "qa.json"
    path.write_text('[{"question":"q","answer":"a"}]', encoding="utf-8")
    assert load_json_file(path) == [{"question": "q", "answer": "a"}]


def test_rejects_malformed_json_with_filename(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("[", encoding="utf-8")
    with pytest.raises(SourceDataError, match="broken.json: malformed JSON"):
        load_json_file(path)

def test_rejects_nonstandard_json_values(tmp_path):
    path = tmp_path / "nonstandard.json"
    path.write_text('[{"question":"q","answer":NaN}]', encoding="utf-8")
    with pytest.raises(SourceDataError, match="non-standard JSON constant"):
        load_json_file(path)


@pytest.mark.parametrize("contents", ['{"records": []}', '[1, {"question":"q"}]'])
def test_rejects_non_array_or_non_object_records(tmp_path, contents):
    path = tmp_path / "invalid.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(SourceDataError):
        load_json_file(path)
