"""Web discovery: publish the probe, search it, fetch the candidates back."""

from __future__ import annotations


class DiscoveryError(RuntimeError):
    """Web discovery could not run, with display-safe fields for the UI.

    `reason` is always derived from a status code or an exception *type*, never
    from a request URL: the SerpApi URL embeds the API key, so echoing a
    transport error verbatim would leak it into the UI and the logs.
    """

    def __init__(self, provider: str, reason: str, timeout: str = "") -> None:
        super().__init__(f"web discovery unavailable ({provider}): {reason}")
        self.provider = provider
        self.reason = reason
        self.timeout = timeout
