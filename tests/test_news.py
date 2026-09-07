"""Tests for cricket news aggregation.

Feeds are never hit here — the parser is exercised against fixture XML so the
suite stays fast and offline.
"""
from __future__ import annotations

import pytest

from backend.services import news as news_mod
from backend.services.news import MAX_LIMIT, _dedupe, _extract_image, _parse_feed, get_news

ESPN_XML = """<?xml version="1.0"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/">
  <channel>
    <item>
      <title>Archer displays new-found durability</title>
      <description>A &lt;b&gt;bold&lt;/b&gt; spell   with   odd spacing</description>
      <coverImages>https://img.example/1.jpg</coverImages>
      <link>https://cricinfo.example/story.html?ex_cid=OTC-RSS</link>
      <url>https://cricinfo.example/canonical-story</url>
      <pubDate>Mon, 07 Sep 2026 16:25:00 GMT</pubDate>
    </item>
    <item>
      <title>No link here</title>
      <link>not-a-url</link>
    </item>
  </channel>
</rss>"""

MEDIA_XML = """<?xml version="1.0"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/">
  <channel>
    <item>
      <title>Santner warns India</title>
      <link>https://ie.example/article</link>
      <description></description>
      <media:thumbnail url="https://img.example/thumb.jpg" />
      <pubDate>Mon, 07 Sep 2026 16:09:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""


@pytest.mark.unit
def test_prefers_canonical_url_over_tracking_link() -> None:
    items = _parse_feed("ESPNcricinfo", ESPN_XML)
    assert items[0].link == "https://cricinfo.example/canonical-story"


@pytest.mark.unit
def test_strips_tags_and_collapses_whitespace() -> None:
    assert _parse_feed("ESPNcricinfo", ESPN_XML)[0].summary == "A bold spell with odd spacing"


@pytest.mark.unit
def test_drops_items_without_a_usable_link() -> None:
    titles = [i.title for i in _parse_feed("ESPNcricinfo", ESPN_XML)]
    assert "No link here" not in titles


@pytest.mark.unit
def test_publish_date_normalised_to_utc_iso() -> None:
    assert _parse_feed("ESPNcricinfo", ESPN_XML)[0].published.startswith("2026-09-07T16:25")


@pytest.mark.unit
@pytest.mark.parametrize(
    "xml,expected",
    [(ESPN_XML, "https://img.example/1.jpg"), (MEDIA_XML, "https://img.example/thumb.jpg")],
)
def test_image_extracted_from_either_convention(xml: str, expected: str) -> None:
    import xml.etree.ElementTree as ET

    assert _extract_image(next(ET.fromstring(xml).iter("item"))) == expected


@pytest.mark.unit
def test_dedupe_collapses_same_story_across_sources() -> None:
    a = _parse_feed("ESPNcricinfo", ESPN_XML)[0]
    near_dup = type(a)(**{**a.__dict__, "title": "Archer displays  NEW-FOUND durability!"})
    assert len(_dedupe([a, near_dup])) == 1


@pytest.mark.unit
def test_a_failing_feed_does_not_lose_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(_client, source, _url):
        if source == "BBC Sport":
            return []  # simulates the failure path
        return _parse_feed(source, ESPN_XML)

    monkeypatch.setattr(news_mod, "_fetch_one", fake_fetch)
    monkeypatch.setattr(news_mod, "_cache", None)
    import asyncio

    result = asyncio.run(get_news(limit=10))
    assert result["items"] and "BBC Sport" not in result["sources"]


@pytest.mark.unit
def test_limit_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(_client, source, _url):
        return _parse_feed(source, ESPN_XML)

    monkeypatch.setattr(news_mod, "_fetch_one", fake_fetch)
    monkeypatch.setattr(news_mod, "_cache", None)
    import asyncio

    assert len(asyncio.run(get_news(limit=9999))["items"]) <= MAX_LIMIT
