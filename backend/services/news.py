"""Cricket news aggregation from public RSS feeds.

Needs no API key, which is why RSS is used rather than a news API: the rest of
the app degrades gracefully when keys are absent, and news should simply always
work. Feeds are fetched concurrently, normalised into one shape, de-duplicated
and cached in memory.

Everything returned here is third-party text. It is passed through unchanged
apart from tag stripping, so callers MUST escape it before rendering as HTML.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable
from xml.etree import ElementTree

import httpx

logger = logging.getLogger("pitchiq.news")

# Public cricket RSS feeds, most authoritative first. Verified reachable from
# Cloud Run on 2026-09-07. Two were dropped: Cricbuzz returns an empty body, and
# Indian Express 403s datacenter IPs (fine from a laptop, blocked from GCP), so
# NDTV and News18 provide the India-focused coverage instead.
FEEDS: tuple[tuple[str, str], ...] = (
    ("ESPNcricinfo", "https://www.espncricinfo.com/rss/content/story/feeds/0.xml"),
    ("NDTV Sports", "https://feeds.feedburner.com/ndtvsports-cricket"),
    ("News18", "https://www.news18.com/rss/cricketnext.xml"),
    ("BBC Sport", "https://feeds.bbci.co.uk/sport/cricket/rss.xml"),
)

FETCH_TIMEOUT_SECONDS = 12
CACHE_TTL_SECONDS = 600  # 10 min — feeds update far slower than our traffic
DEFAULT_LIMIT = 20
MAX_LIMIT = 50
SUMMARY_MAX_CHARS = 220
USER_AGENT = "PitchIQ/1.0 (+https://github.com/prnadar/pitchiq)"

# Media RSS namespace, used by ESPNcricinfo and Indian Express for images.
_MEDIA_NS = {"media": "http://search.yahoo.com/mrss/"}
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NewsItem:
    title: str
    link: str
    source: str
    summary: str
    image: str
    published: str  # ISO 8601 UTC, or "" when the feed omits/mangles it

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "link": self.link,
            "source": self.source,
            "summary": self.summary,
            "image": self.image,
            "published": self.published,
        }


# (fetched_at, items) — module-level cache, per process.
_cache: tuple[float, tuple[NewsItem, ...]] | None = None


def _clean_text(raw: str | None) -> str:
    """Strip tags and collapse whitespace. Not a sanitiser — callers escape."""
    if not raw:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", raw)).strip()


def _parse_published(raw: str | None) -> str:
    if not raw:
        return ""
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return ""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _extract_image(item: ElementTree.Element) -> str:
    """Feeds disagree on where the image lives; try each known location."""
    cover = item.findtext("coverImages")
    if cover and cover.strip():
        return cover.strip()
    for path in ("media:thumbnail", "media:content"):
        el = item.find(path, _MEDIA_NS)
        if el is not None:
            url = el.get("url", "").strip()
            if url:
                return url
    enclosure = item.find("enclosure")
    if enclosure is not None and enclosure.get("type", "").startswith("image/"):
        return enclosure.get("url", "").strip()
    return ""


def _parse_feed(source: str, xml_text: str) -> list[NewsItem]:
    root = ElementTree.fromstring(xml_text)
    items: list[NewsItem] = []
    for item in root.iter("item"):
        title = _clean_text(item.findtext("title"))
        # ESPNcricinfo's <link> carries tracking params; its <url> is canonical.
        link = (item.findtext("url") or item.findtext("link") or "").strip()
        if not title or not link.startswith(("http://", "https://")):
            continue
        summary = _clean_text(item.findtext("description"))
        if len(summary) > SUMMARY_MAX_CHARS:
            summary = summary[:SUMMARY_MAX_CHARS].rsplit(" ", 1)[0] + "…"
        items.append(
            NewsItem(
                title=title,
                link=link,
                source=source,
                summary=summary,
                image=_extract_image(item),
                published=_parse_published(item.findtext("pubDate")),
            )
        )
    return items


async def _fetch_one(client: httpx.AsyncClient, source: str, url: str) -> list[NewsItem]:
    try:
        # follow_redirects matters: ESPNcricinfo 302s to its CDN.
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return _parse_feed(source, resp.text)
    except (httpx.HTTPError, ElementTree.ParseError) as exc:
        # One bad feed must not take out the others.
        logger.warning("news feed failed (%s): %s: %s", source, type(exc).__name__, exc)
        return []


def _dedupe(items: Iterable[NewsItem]) -> list[NewsItem]:
    """Same story often appears in several feeds; keep the first seen."""
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for item in items:
        key = re.sub(r"[^a-z0-9]", "", item.title.lower())[:80]
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


async def get_news(limit: int = DEFAULT_LIMIT, force_refresh: bool = False) -> dict[str, Any]:
    """Aggregated cricket headlines, newest first.

    Always returns a dict; `items` is empty when every feed is unreachable and
    `sources` names the feeds that actually answered.
    """
    global _cache
    limit = max(1, min(int(limit), MAX_LIMIT))
    now = asyncio.get_event_loop().time()

    if not force_refresh and _cache is not None:
        fetched_at, cached = _cache
        if now - fetched_at < CACHE_TTL_SECONDS:
            return {
                "items": [i.as_dict() for i in cached[:limit]],
                "sources": sorted({i.source for i in cached}),
                "cached": True,
            }

    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS, headers=headers) as client:
        results = await asyncio.gather(
            *(_fetch_one(client, source, url) for source, url in FEEDS)
        )

    merged = _dedupe([item for batch in results for item in batch])
    # Undated items sort last rather than pretending to be the oldest.
    merged.sort(key=lambda i: i.published or "", reverse=True)

    if merged:
        _cache = (now, tuple(merged))
    elif _cache is not None:
        # Every feed failed — serve the stale copy rather than an empty page.
        logger.warning("all news feeds failed; serving stale cache")
        stale = _cache[1]
        return {
            "items": [i.as_dict() for i in stale[:limit]],
            "sources": sorted({i.source for i in stale}),
            "cached": True,
            "stale": True,
        }

    return {
        "items": [i.as_dict() for i in merged[:limit]],
        "sources": sorted({i.source for i in merged}),
        "cached": False,
    }


# --- Linking headlines to a fixture ------------------------------------------

# Each side has "strong" terms that only ever mean the cricket team, and "weak"
# ones that are ordinary places. A weak term alone is not evidence: "Punjab"
# matched a Lok Sabha election story, "Delhi" matches any city news. Weak terms
# therefore only count alongside a squad member's name.
TEAM_KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "Mumbai Indians": {"strong": ("mumbai indians",), "weak": ("mumbai",)},
    "Chennai Super Kings": {"strong": ("chennai super kings", "csk"), "weak": ("chennai",)},
    "Royal Challengers Bangalore": {
        "strong": ("royal challengers bangalore", "royal challengers bengaluru", "rcb"),
        "weak": ("bangalore", "bengaluru"),
    },
    "Kolkata Knight Riders": {"strong": ("kolkata knight riders", "kkr"), "weak": ("kolkata",)},
    "Delhi Capitals": {"strong": ("delhi capitals",), "weak": ("delhi",)},
    "Rajasthan Royals": {"strong": ("rajasthan royals",), "weak": ("rajasthan",)},
    "Sunrisers Hyderabad": {"strong": ("sunrisers hyderabad", "sunrisers", "srh"), "weak": ("hyderabad",)},
    "Punjab Kings": {"strong": ("punjab kings", "pbks"), "weak": ("punjab",)},
    "Gujarat Titans": {"strong": ("gujarat titans",), "weak": ("gujarat",)},
    "Lucknow Super Giants": {"strong": ("lucknow super giants", "lsg"), "weak": ("lucknow",)},
}

STRONG_TEAM_WEIGHT = 3
WEAK_TEAM_WEIGHT = 1
PLAYER_MATCH_WEIGHT = 2
# "Patel" or "Singh" alone are too common to attribute to one squad.
MIN_SURNAME_LEN = 5
# Feeds carry evergreen video content — without this, "IPL 2024 highlights"
# surfaces as news about next week's fixture.
RELATED_MAX_AGE_DAYS = 30
DEFAULT_RELATED_LIMIT = 4


def team_terms(team: str) -> dict[str, tuple[str, ...]]:
    """Strong and weak headline terms for a team; unknown teams match on name."""
    known = TEAM_KEYWORDS.get(team)
    if known:
        return known
    return {"strong": (team.lower(),) if team else (), "weak": ()}


def surname_of(player: str) -> str:
    """'MS Dhoni' -> 'dhoni'. Squad data stores initials, headlines use names.

    Returns "" for surnames too short to attribute confidently.
    """
    parts = [p for p in re.split(r"[\s.]+", player.strip()) if p]
    if not parts:
        return ""
    surname = parts[-1].lower()
    return surname if len(surname) >= MIN_SURNAME_LEN else ""


def _mentions(haystack: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", haystack) is not None


def _is_recent(item: NewsItem, now: datetime) -> bool:
    """Undated items are kept: feeds omit pubDate more often than they lie."""
    if not item.published:
        return True
    try:
        published = datetime.fromisoformat(item.published)
    except ValueError:
        return True
    return (now - published).days <= RELATED_MAX_AGE_DAYS


def score_item(
    item: NewsItem,
    strong_terms: Iterable[str],
    weak_terms: Iterable[str],
    surnames: Iterable[str],
) -> int:
    """How strongly a headline relates to a fixture. 0 means unrelated.

    A team must be identified for the story to count. Squad surnames alone are
    not enough, because IPL players also appear for their countries — matching
    on surname pulled in unrelated internationals ("Archer", "Jansen") and
    common names ("Sharma", "Singh") swept up general news.
    """
    haystack = f"{item.title} {item.summary}".lower()
    players = sum(PLAYER_MATCH_WEIGHT for s in surnames if _mentions(haystack, s))
    strong = sum(STRONG_TEAM_WEIGHT for t in strong_terms if _mentions(haystack, t))
    weak = sum(WEAK_TEAM_WEIGHT for t in weak_terms if _mentions(haystack, t))

    if strong:
        return strong + weak + players
    # A place name only counts as the team when a squad member appears with it.
    if weak and players:
        return weak + players
    return 0


async def get_related_news(
    teams: Iterable[str],
    players: Iterable[str] = (),
    limit: int = DEFAULT_RELATED_LIMIT,
) -> dict[str, Any]:
    """Recent headlines that name either side, ranked by how well they fit.

    Reuses the cached feed, so this costs no extra network call. Returns an
    empty list rather than filler when nothing is genuinely related — an IPL
    fixture in the off-season legitimately has no current news.
    """
    await get_news(limit=MAX_LIMIT)
    cached = _cache[1] if _cache is not None else ()

    strong: set[str] = set()
    weak: set[str] = set()
    for team in teams:
        terms = team_terms(team)
        strong.update(terms["strong"])
        weak.update(terms["weak"])
    surnames = {s for s in (surname_of(p) for p in players) if s}

    now = datetime.now(timezone.utc)
    scored = [
        (score_item(i, strong, weak, surnames), i)
        for i in cached
        if _is_recent(i, now)
    ]
    related = [(s, i) for s, i in scored if s > 0]
    # Strongest link first, then most recent.
    related.sort(key=lambda pair: (pair[0], pair[1].published or ""), reverse=True)

    return {
        "items": [i.as_dict() | {"relevance": s} for s, i in related[:limit]],
        "matched_on": sorted(strong | weak | surnames),
    }
