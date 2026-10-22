"""Check Qdrant connectivity without modifying the collection."""

import json
import sys

from app.config import get_settings
from app.runtime import ServiceContainer


def main() -> int:
    settings = get_settings()
    if not settings.qdrant_configured:
        print("Qdrant is not configured. Set QDRANT_URL and QDRANT_API_KEY in .env.")
        return 2
    try:
        health = ServiceContainer(settings).qdrant_health()
    except Exception as error:
        print(f"Qdrant check failed ({type(error).__name__}).")
        return 1
    print(json.dumps(health, indent=2))
    return 0 if health.get("compatible") else 1


if __name__ == "__main__":
    sys.exit(main())
