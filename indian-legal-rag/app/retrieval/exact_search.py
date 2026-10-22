"""A small SQLite index for exact QA and explicitly stated article/section lookup."""

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from app.ingestion.cleaner import normalize_question_key

ARTICLE_QUERY = re.compile(r"\barticle\s+([0-9]+[A-Za-z]?)\b", re.IGNORECASE)
SECTION_QUERY = re.compile(r"\bsection\s+([0-9]+[A-Za-z]?)\b", re.IGNORECASE)


class ExactSearchIndex:
    """Persisted exact-match index synchronized after a successful full ingestion."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("""
            CREATE TABLE IF NOT EXISTS qa_records (
                record_id TEXT PRIMARY KEY,
                normalized_question TEXT NOT NULL,
                article_number TEXT,
                section_number TEXT,
                law_name TEXT NOT NULL,
                source_file TEXT NOT NULL,
                payload TEXT NOT NULL
            )
        """)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_qa_question ON qa_records(normalized_question)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_qa_article ON qa_records(article_number)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_qa_section ON qa_records(section_number)"
        )
        return connection

    def sync(self, records: list[dict[str, Any]]) -> None:
        """Atomically replace the local exact index after vector upserts complete."""
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM qa_records")
            connection.executemany(
                """
                INSERT INTO qa_records
                (record_id, normalized_question, article_number, section_number,
                 law_name, source_file, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record["record_id"],
                        normalize_question_key(record["question"]),
                        record.get("article_number"),
                        record.get("section_number"),
                        record["law_name"],
                        record["source_file"],
                        json.dumps(record, ensure_ascii=False),
                    )
                    for record in records
                ],
            )

    def search(
        self,
        question: str,
        law_name: str | None = None,
        source_file: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Find exact normalized questions and only explicit article/section matches."""
        clauses: list[str] = []
        params: list[object] = []
        if law_name:
            clauses.append("law_name = ?")
            params.append(law_name)
        if source_file:
            clauses.append("source_file = ?")
            params.append(source_file)
        scope = f" AND {' AND '.join(clauses)}" if clauses else ""
        candidates: list[tuple[str, str]] = []
        exact_params = [normalize_question_key(question), *params, limit]
        with closing(self._connect()) as connection:
            exact_rows = connection.execute(
                f"SELECT payload FROM qa_records WHERE normalized_question = ?{scope} LIMIT ?",
                exact_params,
            ).fetchall()
            candidates.extend((str(row["payload"]), "exact_question") for row in exact_rows)
            article_match = ARTICLE_QUERY.search(question)
            section_match = SECTION_QUERY.search(question)
            if article_match:
                rows = connection.execute(
                    f"SELECT payload FROM qa_records WHERE article_number = ?{scope} LIMIT ?",
                    [article_match.group(1), *params, limit],
                ).fetchall()
                candidates.extend((str(row["payload"]), "explicit_article") for row in rows)
            if section_match:
                rows = connection.execute(
                    f"SELECT payload FROM qa_records WHERE section_number = ?{scope} LIMIT ?",
                    [section_match.group(1), *params, limit],
                ).fetchall()
                candidates.extend((str(row["payload"]), "explicit_section") for row in rows)
        seen: set[str] = set()
        results = []
        for payload_json, method in candidates:
            record = json.loads(payload_json)
            if record["record_id"] in seen:
                continue
            seen.add(record["record_id"])
            results.append({
                **record,
                "retrieval_method": method,
                "rank": len(results) + 1,
                "score": 1.0,
            })
            if len(results) >= limit:
                break
        return results
