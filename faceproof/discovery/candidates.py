"""Candidate parsing, deduplication and investigation ranking.

A candidate is content the reverse image search considers *visually related* to
the probe. It is deliberately not treated as the same person here - that
decision belongs to `faceproof.matching.verifier`.

Ranking in this module answers only "which candidates are worth the expensive
face comparison, and in what order?". It is a retrieval heuristic. Whether a
candidate shows the same person is decided nowhere but the matching layer, on
ArcFace cosine distance against the configured threshold, and a high priority
score is never evidence of identity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from faceproof.discovery.normalize import canonical_url

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

# Investigation priority, not identity confidence. Every weight is a property
# of the *search result*, never of the face in it:
#
#   source            a social post is the evidence this project is looking for,
#                     and it carries an author and a date a reviewer can follow
#   cached thumbnail  the provider's own copy; an origin-only image URL is the
#                     one that answers 403 and burns budget on DOWNLOAD_FAILED
#   search rank       the provider's own ordering, decaying to nothing by
#                     RANK_HORIZON so it only ever breaks ties between the above
#
# Deduplication is the duplicate penalty: a duplicate is merged into its
# original before ranking, so it cannot consume the budget twice.
SOURCE_PRIORITY = {"social": 3.0, "web": 0.0}
CACHED_THUMBNAIL_PRIORITY = 1.0
RANK_PRIORITY = 1.0
RANK_HORIZON = 25


@dataclass(frozen=True)
class Candidate:
    """One visually related result returned by the reverse image search.

    `page_url` and `image_url` are always the provider's originals - they are
    what gets fetched and what the evidence records. The canonical forms beside
    them exist only to compare two candidates.
    """

    title: str
    page_url: str
    image_url: str
    source: str
    position: int
    is_social: bool
    canonical_url: str = ""
    canonical_image_url: str = ""
    priority_score: float = 0.0
    priority_reasons: Tuple[str, ...] = ()
    # original page URLs of results merged into this one, in the order merged
    duplicates: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        """The identity of this result.

        Ranking fields are deliberately absent: this feeds the evidence-facing
        shape, where a retrieval heuristic has no business.
        """
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


def source_priority(is_social: bool) -> float:
    """Weight for where a candidate was found."""
    return SOURCE_PRIORITY["social" if is_social else "web"]


def priority(
    is_social: bool, cached_thumbnail: bool, position: int
) -> Tuple[float, Tuple[str, ...]]:
    """Score one candidate for investigation order, and say why.

    Deterministic and additive: the same result always scores the same, so the
    same search response always produces the same investigation order.
    """
    score = source_priority(is_social)
    reasons: List[str] = ["social source"] if is_social else []

    if cached_thumbnail:
        score += CACHED_THUMBNAIL_PRIORITY
        reasons.append("cached thumbnail")

    rank_signal = max(0.0, 1.0 - (max(position, 1) - 1) / RANK_HORIZON)
    if rank_signal:
        score += RANK_PRIORITY * rank_signal
        reasons.append(f"search rank {position}")

    return round(score, 4), tuple(reasons)


def _parse_one(index: int, item: Dict[str, Any]) -> Optional[Candidate]:
    """Build one Candidate, or None when the entry cannot be investigated."""
    page_url = item.get("link") or ""
    thumbnail = item.get("thumbnail") or ""
    image_url = thumbnail or item.get("image") or ""
    if not page_url or not image_url:
        return None  # unverifiable and unevidencable: there is nothing to check

    source = item.get("source") or ""
    position = item.get("position", index + 1)
    is_social = is_social_url(page_url) or is_social_url(source)
    score, reasons = priority(is_social, bool(thumbnail), position)
    return Candidate(
        title=item.get("title") or "",
        page_url=page_url,
        image_url=image_url,
        source=source,
        position=position,
        is_social=is_social,
        canonical_url=canonical_url(page_url),
        canonical_image_url=canonical_url(image_url),
        priority_score=score,
        priority_reasons=reasons,
    )


def _better(one: Candidate, other: Candidate) -> Candidate:
    """The member of a duplicate group worth keeping."""
    return max((one, other), key=lambda c: (c.priority_score, -c.position))


def deduplicate(candidates: Sequence[Candidate]) -> List[Candidate]:
    """Merge results that are exactly the same page, or exactly the same image.

    Two keys, both exact after canonicalization and nothing softer: the page
    URL, then the image URL. No content hashing - that would mean downloading
    an image to decide whether to download it - and no perceptual hashing,
    which is a similarity guess and has no place in a system whose whole claim
    is that identity is decided by one measurable distance.

    The survivor is the higher-priority member, so merging a tracking-tagged
    copy of a social post into a plain one never costs the better source. Every
    URL merged away is kept on the survivor, so provenance is preserved rather
    than deleted.
    """
    kept: List[Candidate] = []
    where: Dict[str, int] = {}  # canonical page/image URL -> index into `kept`

    for candidate in candidates:
        keys = [key for key in (candidate.canonical_url, candidate.canonical_image_url) if key]
        found = next((where[key] for key in keys if key in where), None)
        if found is None:
            kept.append(candidate)
            for key in keys:
                where.setdefault(key, len(kept) - 1)
            continue

        original = kept[found]
        survivor = _better(original, candidate)
        loser = candidate if survivor is original else original
        kept[found] = replace(
            survivor,
            duplicates=original.duplicates + candidate.duplicates + (loser.page_url,),
        )
        # the loser's keys must resolve here too, or its own twin would survive
        for key in keys:
            where.setdefault(key, found)

    return kept


def rank(candidates: Sequence[Candidate]) -> List[Candidate]:
    """Order candidates by investigation priority, highest first.

    Position and page URL break ties, so the order is total and stable: the
    same response always yields the same order, on any machine.
    """
    return sorted(
        candidates,
        key=lambda candidate: (-candidate.priority_score, candidate.position, candidate.page_url),
    )


def discovered_count(candidates: Sequence[Candidate]) -> int:
    """How many results the search returned, duplicates included."""
    return sum(1 + len(candidate.duplicates) for candidate in candidates)


def parse_visual_matches(payload: Dict[str, Any]) -> List[Candidate]:
    """Turn a reverse-search response into ranked, deduplicated candidates.

    Parse, normalize, deduplicate, rank - in that order, and all of it before a
    single face is compared, so the candidate budget is spent on the most
    promising results rather than on whatever the provider listed first.

    Entries missing a page URL or an image URL are dropped: they cannot be
    verified or evidenced. Everything else survives - a candidate below the
    budget is uninvestigated, which is not the same as rejected.

    ponytail: the image URL is whatever thumbnail the provider hands back, so
    matching runs on low-resolution crops. Resolve the highest-quality public
    image per candidate if false negatives become a problem.
    """
    parsed = [
        _parse_one(index, item) for index, item in enumerate(payload.get("visual_matches") or [])
    ]
    return rank(deduplicate([candidate for candidate in parsed if candidate is not None]))
