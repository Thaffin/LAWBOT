import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_FILES = (
    "constitution_qa.json",
    "ipc_qa.json",
    "crpc_qa.json",
)
PROCESSING_VERSION = "1.0.0"
TEXT_FIELD_NAMES = {
    "text", "content", "body", "question", "answer", "response",
    "passage", "description", "provision", "explanation"
}
IDENTIFIER_FIELD_NAMES = {
    "id", "document_id", "record_id", "question_id", "section_id",
    "article_id", "source_id"
}


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def normalize_string(value):
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = value.split("\n")
    cleaned_lines = []
    for line in lines:
        line = re.sub(r"[^\S\n]+", " ", line).strip()
        cleaned_lines.append(line)
    value = "\n".join(cleaned_lines)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def normalize_value(value):
    if isinstance(value, str):
        return normalize_string(value)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_value(item) for key, item in value.items()}
    return value


def value_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def walk_values(value, path="$", key_stats=None, type_counts=None, strings=None):
    if key_stats is None:
        key_stats = defaultdict(lambda: {
            "occurrences": 0,
            "value_types": Counter(),
            "empty_values": 0,
            "example_values": []
        })
    if type_counts is None:
        type_counts = Counter()
    if strings is None:
        strings = []

    type_counts[value_type(value)] += 1

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            stats = key_stats[child_path]
            stats["occurrences"] += 1
            stats["value_types"][value_type(child)] += 1
            if child is None or child == "" or child == [] or child == {}:
                stats["empty_values"] += 1
            if len(stats["example_values"]) < 3:
                example = child
                if isinstance(example, (dict, list)):
                    example = f"<{value_type(example)}>"
                elif isinstance(example, str) and len(example) > 160:
                    example = example[:157] + "..."
                if example not in stats["example_values"]:
                    stats["example_values"].append(example)
            walk_values(child, child_path, key_stats, type_counts, strings)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk_values(child, f"{path}[{index}]", key_stats, type_counts, strings)
    elif isinstance(value, str):
        strings.append(value)

    return key_stats, type_counts, strings


def count_words(text):
    return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))


def top_level_records(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list) and key.lower() in {
                "data", "records", "items", "questions", "qa", "examples"
            }:
                return value
        return [data]
    return []


def canonical_hash(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inspect_records(records):
    exact_hashes = {}
    duplicate_records = []
    empty_text_fields = []
    missing_identifiers = []
    suspiciously_short = []

    for index, record in enumerate(records):
        record_number = index + 1
        if isinstance(record, (dict, list)):
            digest = canonical_hash(record)
            if digest in exact_hashes:
                duplicate_records.append({
                    "record_number": record_number,
                    "first_occurrence_record_number": exact_hashes[digest],
                    "sha256": digest
                })
            else:
                exact_hashes[digest] = record_number

        if isinstance(record, dict):
            lower_keys = {str(key).lower(): key for key in record.keys()}
            for field_name in TEXT_FIELD_NAMES:
                if field_name in lower_keys:
                    original_key = lower_keys[field_name]
                    field_value = record[original_key]
                    if field_value is None or (
                        isinstance(field_value, str) and not field_value.strip()
                    ):
                        empty_text_fields.append({
                            "record_number": record_number,
                            "field": str(original_key)
                        })

            if not any(name in lower_keys for name in IDENTIFIER_FIELD_NAMES):
                missing_identifiers.append(record_number)

            text_parts = []
            for key, value in record.items():
                if str(key).lower() in TEXT_FIELD_NAMES and isinstance(value, str):
                    text_parts.append(value.strip())
            if text_parts and sum(len(part) for part in text_parts) < 20:
                suspiciously_short.append({
                    "record_number": record_number,
                    "combined_text_characters": sum(len(part) for part in text_parts)
                })

    return {
        "exact_duplicate_records": duplicate_records,
        "empty_text_fields": empty_text_fields,
        "records_without_recognized_identifier": missing_identifiers,
        "suspiciously_short_text_records": suspiciously_short
    }


def analyze_json(data, raw_text, raw_bytes, source_path):
    key_stats, type_counts, strings = walk_values(data)
    records = top_level_records(data)
    all_keys = sorted({
        str(key)
        for item in (data if isinstance(data, list) else [data])
        if isinstance(item, dict)
        for key in item.keys()
    })

    key_report = {}
    for path, stats in sorted(key_stats.items()):
        key_report[path] = {
            "occurrences": stats["occurrences"],
            "value_types": dict(stats["value_types"]),
            "empty_values": stats["empty_values"],
            "example_values": stats["example_values"]
        }

    non_empty_strings = [item for item in strings if item.strip()]
    string_characters = sum(len(item) for item in strings)
    string_words = sum(count_words(item) for item in strings)

    return {
        "source_file": source_path.name,
        "source_path": str(source_path),
        "file_size_bytes": len(raw_bytes),
        "file_size_kib": round(len(raw_bytes) / 1024, 3),
        "raw_file_characters": len(raw_text),
        "raw_file_words": count_words(raw_text),
        "top_level_json_type": value_type(data),
        "top_level_items": len(data) if isinstance(data, (dict, list)) else None,
        "record_count_for_inspection": len(records),
        "top_level_keys": all_keys,
        "all_nested_key_paths": key_report,
        "value_type_counts": dict(type_counts),
        "string_value_count": len(strings),
        "non_empty_string_value_count": len(non_empty_strings),
        "string_value_characters_total": string_characters,
        "string_value_words_total": string_words,
        "average_string_length_characters": round(
            string_characters / len(strings), 2
        ) if strings else 0,
        "quality_checks": inspect_records(records)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate, inspect, normalize and report on Indian legal QA JSON files."
    )
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/cleaned")
    parser.add_argument("--report-dir", default="reports/json_cleaning")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = utc_timestamp()
    combined_report = {
        "report_type": "indian_legal_json_validation_and_cleaning",
        "processing_version": PROCESSING_VERSION,
        "run_timestamp_utc": run_timestamp,
        "python_version": sys.version,
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "files": [],
        "missing_files": [],
        "errors": []
    }

    for filename in EXPECTED_FILES:
        source_path = input_dir / filename
        if not source_path.exists():
            combined_report["missing_files"].append(filename)
            print(f"[MISSING] {source_path}")
            continue

        try:
            raw_bytes = source_path.read_bytes()
            raw_text = raw_bytes.decode("utf-8-sig")
            data = json.loads(raw_text)

            normalized_data = normalize_value(data)
            cleaned_path = output_dir / filename
            cleaned_text = json.dumps(
                normalized_data,
                ensure_ascii=False,
                indent=2,
                allow_nan=False
            ) + "\n"
            cleaned_path.write_text(cleaned_text, encoding="utf-8")

            file_report = analyze_json(
                data=data,
                raw_text=raw_text,
                raw_bytes=raw_bytes,
                source_path=source_path
            )
            normalized_bytes = cleaned_text.encode("utf-8")
            file_report["cleaned_file"] = str(cleaned_path)
            file_report["cleaned_file_size_bytes"] = len(normalized_bytes)
            file_report["cleaned_file_characters"] = len(cleaned_text)
            file_report["cleaned_file_words"] = count_words(cleaned_text)
            file_report["cleaning_actions"] = [
                "Unicode normalized to NFC",
                "Line endings normalized to LF",
                "Repeated horizontal whitespace reduced to one space",
                "Leading and trailing whitespace removed from string values",
                "Runs of three or more newline characters reduced to two",
                "JSON serialized with UTF-8, indentation and stable key order disabled"
            ]
            file_report["notes"] = [
                "Original input was not modified",
                "Exact duplicates and quality concerns are reported, not deleted automatically",
                "Legal numbering, punctuation, quotation marks, field names and record order are preserved",
                "Schema-specific required-field validation must be finalized after the real files are inspected"
            ]
            combined_report["files"].append(file_report)
            print(
                f"[OK] {filename}: {file_report['record_count_for_inspection']} "
                f"records inspected; output={cleaned_path}"
            )

        except UnicodeDecodeError as error:
            message = f"{filename}: not valid UTF-8 text: {error}"
            combined_report["errors"].append(message)
            print(f"[ERROR] {message}")
        except json.JSONDecodeError as error:
            message = (
                f"{filename}: invalid JSON at line {error.lineno}, "
                f"column {error.colno}: {error.msg}"
            )
            combined_report["errors"].append(message)
            print(f"[ERROR] {message}")
        except (OSError, ValueError, TypeError) as error:
            message = f"{filename}: {type(error).__name__}: {error}"
            combined_report["errors"].append(message)
            print(f"[ERROR] {message}")

    report_path = report_dir / "json_cleaning_report.json"
    summary_path = report_dir / "json_cleaning_summary.txt"
    report_path.write_text(
        json.dumps(combined_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    summary_lines = [
        "INDIAN LEGAL JSON VALIDATION AND CLEANING REPORT",
        f"Run timestamp (UTC): {run_timestamp}",
        f"Processing version: {PROCESSING_VERSION}",
        f"Python version: {sys.version.split()[0]}",
        f"Input directory: {input_dir}",
        f"Output directory: {output_dir}",
        ""
    ]

    for file_report in combined_report["files"]:
        checks = file_report["quality_checks"]
        summary_lines.extend([
            f"FILE: {file_report['source_file']}",
            f"  Raw size: {file_report['file_size_bytes']} bytes ({file_report['file_size_kib']} KiB)",
            f"  Raw characters: {file_report['raw_file_characters']}",
            f"  Raw words: {file_report['raw_file_words']}",
            f"  Top-level JSON type: {file_report['top_level_json_type']}",
            f"  Top-level keys: {', '.join(file_report['top_level_keys']) or '(none)'}",
            f"  Records inspected: {file_report['record_count_for_inspection']}",
            f"  String values: {file_report['string_value_count']}",
            f"  String-value characters: {file_report['string_value_characters_total']}",
            f"  String-value words: {file_report['string_value_words_total']}",
            f"  Exact duplicate records flagged: {len(checks['exact_duplicate_records'])}",
            f"  Empty text fields flagged: {len(checks['empty_text_fields'])}",
            f"  Records without recognized identifier: {len(checks['records_without_recognized_identifier'])}",
            f"  Suspiciously short text records: {len(checks['suspiciously_short_text_records'])}",
            f"  Cleaned output: {file_report['cleaned_file']}",
            ""
        ])

    summary_lines.extend([
        f"Missing files: {', '.join(combined_report['missing_files']) or '(none)'}",
        f"Errors: {len(combined_report['errors'])}",
        *[f"  - {error}" for error in combined_report["errors"]],
        "",
        "Important: quality flags are review signals, not proof that a legal record is invalid.",
        "Inspect all schema paths and manually verify a sample against authoritative legal sources."
    ])
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[REPORT] {report_path}")
    print(f"[SUMMARY] {summary_path}")

    if combined_report["errors"]:
        raise SystemExit(1)
    if combined_report["missing_files"] and not args.allow_missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
