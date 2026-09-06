"""URL canonicalization, used only to decide whether two candidates are the same.

A reverse image search can return the same page twice through different URLs: a
share link carrying a tracking parameter, a host in different casing, an
explicit default port. Comparing the canonical forms collapses those without
guessing.

Nothing here rewrites what a reviewer sees. The candidate keeps its original
`page_url` and `image_url`, and those are what the evidence records - the
canonical form is a comparison key beside them, never a replacement.
"""

from __future__ import annotations

import re
from typing import List, Tuple
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit, urlunsplit

# Removed before two URLs are compared. Every entry is analytics or share
# attribution: it records how a visitor arrived, never *what* they arrived at.
# Anything that can select content stays - `?id=123` and `?id=456` are
# different posts, `?v=abc` is a different video, `?format=webp` is different
# image bytes, and merging those would silently drop a candidate.
TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "utm_name",
        "utm_reader",
        "fbclid",
        "gclid",
        "dclid",
        "gbraid",
        "wbraid",
        "msclkid",
        "yclid",
        "twclid",
        "igshid",
        "igsh",
        "mc_cid",
        "mc_eid",
        "_ga",
        "_gl",
        "ref_src",
        "ref_url",
    }
)

DEFAULT_PORTS = {"http": "80", "https": "443", "ftp": "21"}

_PERCENT_ESCAPE = re.compile(r"%[0-9a-fA-F]{2}")


def _host(parts: SplitResult) -> str:
    """Lowercased host with any default port dropped, userinfo left alone.

    Userinfo is case sensitive, so the whole netloc cannot simply be lowered.
    """
    try:
        host, port = (parts.hostname or ""), parts.port
    except ValueError:  # an unparsable port, e.g. http://host:notaport/
        return parts.netloc.lower()
    userinfo, _, _ = parts.netloc.rpartition("@")
    default = DEFAULT_PORTS.get(parts.scheme.lower())
    suffix = "" if port is None or str(port) == default else f":{port}"
    return f"{userinfo}@{host}{suffix}" if userinfo else f"{host}{suffix}"


def _query(query: str) -> str:
    """Drop tracking parameters and sort the rest.

    Sorting is safe here because the result is only ever compared, never
    fetched: `?a=1&b=2` and `?b=2&a=1` are the same page to every server that
    matters, and the original URL is what gets requested.
    """
    kept: List[Tuple[str, str]] = [
        (key, value)
        for key, value in parse_qsl(query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
    ]
    return urlencode(sorted(kept))


def canonical_url(url: str) -> str:
    """A comparable form of `url`, or the input unchanged if it cannot be parsed.

    Applied identically to page and image URLs: an image CDN's query string
    (`?format=webp&name=large`) selects different bytes, so it is content and
    survives, exactly like a page's `?id=`.

    Normalized: scheme and host casing, default ports, percent-escape casing,
    one trailing slash, tracking parameters (`TRACKING_PARAMS`), parameter
    order, and fragments other than a hashbang route.
    """
    if not url or not url.strip():
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url
    if not parts.netloc:
        return url  # relative or malformed: there is nothing safe to normalize

    path = _PERCENT_ESCAPE.sub(lambda found: found.group().upper(), parts.path)
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    # a fragment is client-side, except where a hashbang carries the route
    fragment = parts.fragment if parts.fragment.startswith("!") else ""
    return urlunsplit((parts.scheme.lower(), _host(parts), path, _query(parts.query), fragment))
