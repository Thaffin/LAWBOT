"""Load source JSON files independently and validate their outer structure."""

import json
import logging
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT

logger = logging.getLogger(__name__)
SOURCE_FILES = ("constitution_qa.json", "ipc_qa.json", "crpc_qa.json")


class SourceDataError(ValueError):
    """Raised when a source file cannot be parsed or has an invalid structure."""


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def load_json_file(path: str | Path) -> list[dict[str, Any]]:
    """Return a JSON array of objects, with errors identifying only the filename."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle, parse_constant=_reject_nonstandard_constant)
    except json.JSONDecodeError as error:
        raise SourceDataError(
            f"{source.name}: malformed JSON at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error
    except UnicodeDecodeError as error:
        raise SourceDataError(f"{source.name}: input is not valid UTF-8") from error
    except ValueError as error:
        raise SourceDataError(f"{source.name}: malformed JSON ({error})") from error
    except OSError as error:
        raise SourceDataError(
            f"{source.name}: unable to read source file ({type(error).__name__})"
        ) from error
    if not isinstance(data, list):
        raise SourceDataError(f"{source.name}: expected a top-level JSON array")
    invalid = [index for index, item in enumerate(data, start=1) if not isinstance(item, dict)]
    if invalid:
        raise SourceDataError(
            f"{source.name}: expected objects in the JSON array; "
            f"invalid record positions include {invalid[:10]}"
        )
    return data


def load_sources(input_dir: str | Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Read each configured source independently; missing/invalid files are reported."""
    directory = Path(input_dir) if input_dir else PROJECT_ROOT / "data" / "raw"
    loaded: dict[str, list[dict[str, Any]]] = {}
    for filename in SOURCE_FILES:
        try:
            loaded[filename] = load_json_file(directory / filename)
        except SourceDataError as error:
            logger.error("%s", error)
    return loaded
