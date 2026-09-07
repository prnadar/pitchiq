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


# --- Linking headlines to a fixture ------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402

from backend.services.news import (  # noqa: E402
    NewsItem, RELATED_MAX_AGE_DAYS, _is_recent, score_item, surname_of, team_terms,
)


def _item(title: str, summary: str = "", published: str = "") -> NewsItem:
    return NewsItem(title=title, link="https://e.example/x", source="Test",
                    summary=summary, image="", published=published)


def _score(title: str, teams=("Mumbai Indians",), players=()) -> int:
    strong: set[str] = set()
    weak: set[str] = set()
    for t in teams:
        terms = team_terms(t)
        strong.update(terms["strong"])
        weak.update(terms["weak"])
    surnames = {s for s in (surname_of(p) for p in players) if s}
    return score_item(_item(title), strong, weak, surnames)


@pytest.mark.unit
@pytest.mark.parametrize("player,expected", [
    ("MS Dhoni", "dhoni"), ("A Chopra", "chopra"), ("R Sharma", "sharma"),
    ("V Kohli", "kohli"), ("S Iyer", ""),  # too short to attribute
    ("", ""),
])
def test_surname_extraction(player: str, expected: str) -> None:
    assert surname_of(player) == expected


@pytest.mark.unit
def test_named_team_scores() -> None:
    assert _score("Mumbai Indians seal a thriller") > 0


@pytest.mark.unit
def test_surname_alone_is_not_enough() -> None:
    """IPL players also play internationals — a surname must not imply the team."""
    assert _score("Archer displays new-found durability", players=("J Archer",)) == 0


@pytest.mark.unit
def test_place_name_alone_is_not_enough() -> None:
    """Regression: 'Punjab' matched a Lok Sabha election story."""
    assert _score("Yuvraj Singh will not contest Lok Sabha polls",
                  teams=("Punjab Kings",)) == 0


@pytest.mark.unit
def test_place_name_counts_when_a_squad_member_appears() -> None:
    assert _score("Punjab hand Wadhera a new role",
                  teams=("Punjab Kings",), players=("N Wadhera",)) > 0


@pytest.mark.unit
def test_squad_member_boosts_a_team_story_above_a_bare_mention() -> None:
    bare = _score("Mumbai Indians confirm fixtures")
    with_player = _score("Mumbai Indians confirm Sharma is fit",
                         players=("R Sharma",))
    assert with_player > bare


@pytest.mark.unit
def test_stale_items_are_excluded() -> None:
    """Feeds carry evergreen video; 'IPL 2024 highlights' is not news."""
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=RELATED_MAX_AGE_DAYS + 5)).isoformat()
    fresh = (now - timedelta(days=1)).isoformat()
    assert not _is_recent(_item("IPL 2024 highlights", published=old), now)
    assert _is_recent(_item("Today's report", published=fresh), now)


@pytest.mark.unit
def test_undated_items_are_kept() -> None:
    """Feeds omit pubDate more often than they misreport it."""
    assert _is_recent(_item("No date"), datetime.now(timezone.utc))
