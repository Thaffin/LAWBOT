"""Validate the configured QA JSON inputs without changing them."""

import json
import sys

from app.ingestion.cleaner import validate_sources


def main() -> int:
    report = validate_sources()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
