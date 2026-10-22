"""Clean QA datasets into data/cleaned and write a report."""

import json
import sys

from app.ingestion.cleaner import clean_dataset


def main() -> int:
    report = clean_dataset()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
