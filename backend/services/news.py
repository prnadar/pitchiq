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

# Public cricket RSS feeds, most authoritative first. Verified reachable
# 2026-09-07; Cricbuzz was dropped because its feed returns an empty body.
FEEDS: tuple[tuple[str, str], ...] = (
    ("ESPNcricinfo", "https://www.espncricinfo.com/rss/content/story/feeds/0.xml"),
    ("Indian Express", "https://indianexpress.com/section/sports/cricket/feed/"),
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
