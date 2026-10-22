"""Configurable OpenAI-compatible hosted inference client."""

import re
from typing import Any

import httpx

from app.config import Settings

RECORD_ID_CITATION = re.compile(r"\[([0-9a-f]{64})\]")


class LLMConfigurationError(RuntimeError):
    """Raised when required generation settings are not configured."""


class LLMServiceError(RuntimeError):
    """Raised for safe, user-facing generation failures."""


class LLMClient:
    """Call an OpenAI-compatible chat completion endpoint with bounded timeouts."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate(self, messages: list[dict[str, str]]) -> str:
        base_url, model = self.settings.llm_base_url, self.settings.llm_model
        if not base_url or not model:
            raise LLMConfigurationError(
                "Answer generation is not configured; set LLM_BASE_URL and LLM_MODEL."
            )
        base_url = base_url.rstrip("/")
        endpoint = (
            base_url if base_url.endswith("/chat/completions")
            else f"{base_url}/chat/completions"
        )
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        request_body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
                response = await client.post(endpoint, headers=headers, json=request_body)
                response.raise_for_status()
                data = response.json()
        except (httpx.TimeoutException, httpx.ConnectError) as error:
            raise LLMServiceError("The configured language-model service is unavailable.") from error
        except httpx.HTTPStatusError as error:
            raise LLMServiceError(
                f"The language-model service returned HTTP {error.response.status_code}."
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            raise LLMServiceError("The language-model service returned an invalid response.") from error
        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMServiceError("The language-model response did not contain an answer.") from error
        if not isinstance(answer, str) or not answer.strip():
            raise LLMServiceError("The language-model response did not contain an answer.")
        return answer.strip()


def validate_citations(answer: str, evidence: list[dict[str, Any]]) -> None:
    """Reject citation identifiers not present in the evidence supplied to generation."""
    allowed = {str(item["record_id"]) for item in evidence}
    cited = set(RECORD_ID_CITATION.findall(answer))
    if cited - allowed:
        raise ValueError("Generated response cited a source identifier not in retrieved evidence.")
