"""Reverse image search.

Nothing here is hardcoded: the full response is returned to the caller and
persisted as search proof, after credential-shaped fields are removed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from faceproof.discovery import DiscoveryError
from faceproof.discovery.candidates import Candidate, parse_visual_matches
from faceproof.discovery.retrieval import SEARCH_TIMEOUT

SERPAPI_ENDPOINT = "https://serpapi.com/search"
DEFAULT_ENGINE = "google_lens"
SEARCH_ENGINE_ID = "serpapi/google_lens"
TIMEOUT_NOTE = f"{SEARCH_TIMEOUT[0]}s connect / {SEARCH_TIMEOUT[1]}s read"


def redact(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Strip anything credential-shaped before the response is written to disk."""
    cleaned = dict(payload)
    parameters = cleaned.get("search_parameters")
    if isinstance(parameters, dict):
        cleaned["search_parameters"] = {
            key: ("<redacted>" if "key" in key.lower() else value)
            for key, value in parameters.items()
        }
    cleaned.pop("search_metadata", None)
    return cleaned


def _failure_reason(error: requests.RequestException) -> str:
    """Describe a failed search using the status code and nothing else.

    The request URL carries `api_key`, and it is embedded in both `str(error)`
    and `error.request.url` - so neither may ever reach the caller.
    """
    if isinstance(error, requests.Timeout):
        return f"reverse image search timed out after {SEARCH_TIMEOUT[1]}s"
    status = getattr(error.response, "status_code", None)
    if status in (401, 403):
        return "reverse image search rejected the credentials (check SERPAPI_API_KEY)"
    if status == 429:
        return "reverse image search rate limit reached (HTTP 429) - wait and retry"
    if status is not None:
        return f"reverse image search failed with HTTP {status}"
    return f"reverse image search could not reach {SERPAPI_ENDPOINT}"


def reverse_search(
    image_url: str,
    api_key: Optional[str] = None,
    engine: str = DEFAULT_ENGINE,
) -> Tuple[List[Candidate], Dict[str, Any]]:
    """Run the reverse image search. Returns (candidates, redacted raw response)."""
    api_key = api_key or os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        raise DiscoveryError(
            SEARCH_ENGINE_ID, "SERPAPI_API_KEY is not set (copy .env.example to .env)"
        )

    try:
        response = requests.get(
            SERPAPI_ENDPOINT,
            params={"engine": engine, "url": image_url, "api_key": api_key},
            timeout=SEARCH_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise DiscoveryError(SEARCH_ENGINE_ID, _failure_reason(error), TIMEOUT_NOTE) from None

    payload = response.json()
    if payload.get("error"):
        raise DiscoveryError(SEARCH_ENGINE_ID, str(payload["error"]))

    return parse_visual_matches(payload), redact(payload)
