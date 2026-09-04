"""Candidate parsing and ranking.

A candidate is content the reverse image search considers *visually related* to
the probe. It is deliberately not treated as the same person here - that
decision belongs to `faceproof.matching.verifier`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

SOCIAL_DOMAINS = (
    "instagram.com",
    "x.com",
    "twitter.com",
    "linkedin.com",
    "facebook.com",
    "fb.com",
    "reddit.com",
    "tiktok.com",
    "youtube.com",
    "pinterest.com",
    "threads.net",
    "flickr.com",
    "vk.com",
    "tumblr.com",
    "mastodon.social",
)


@dataclass(frozen=True)
class Candidate:
    """One visually related result returned by the reverse image search."""

    title: str
    page_url: str
    image_url: str
    source: str
    position: int
    is_social: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "page_url": self.page_url,
            "image_url": self.image_url,
            "source": self.source,
            "position": self.position,
            "is_social": self.is_social,
        }


def is_social_url(url: str) -> bool:
    """True when the URL points at a known social network."""
    lowered = (url or "").lower()
    return any(domain in lowered for domain in SOCIAL_DOMAINS)


def parse_visual_matches(payload: Dict[str, Any]) -> List[Candidate]:
    """Turn a reverse-search response into Candidate objects.

    Social results are ranked first so the verification stage spends its budget
    where a real social post is most likely to be. Entries missing a page URL or
    an image URL are dropped: they cannot be verified or evidenced.

    ponytail: the image URL is whatever thumbnail the provider hands back, so
    matching runs on low-resolution crops. Resolve the highest-quality public
    image per candidate if false negatives become a problem.
    """
    candidates: List[Candidate] = []
    for index, item in enumerate(payload.get("visual_matches") or []):
        page_url = item.get("link") or ""
        image_url = item.get("thumbnail") or item.get("image") or ""
        if not page_url or not image_url:
            continue
        candidates.append(
            Candidate(
                title=item.get("title") or "",
                page_url=page_url,
                image_url=image_url,
                source=item.get("source") or "",
                position=item.get("position", index + 1),
                is_social=is_social_url(page_url) or is_social_url(item.get("source") or ""),
            )
        )
    candidates.sort(key=lambda candidate: (not candidate.is_social, candidate.position))
    return candidates
