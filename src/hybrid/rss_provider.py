"""RSS/Atom feed aggregation for PhaseBreak Hybrid RAG engine.

Replaces brittle DuckDuckGo HTML scraping with structured RSS feeds.
Falls back gracefully — never raises, always returns [].
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger()

# Lazy import — feedparser may not be installed
_feedparser = None


def _get_feedparser():
    global _feedparser
    if _feedparser is None:
        import feedparser

        _feedparser = feedparser
    return _feedparser


class RSSFeedProvider:
    """RSS/Atom feed aggregation with TTL cache.

    Loads feed config from configs/rss_feeds.yaml.
    Searches by ticker mention in title, description, or tags.
    """

    def __init__(
        self,
        feeds_config_path: str = "configs/rss_feeds.yaml",
        cache_ttl_seconds: int = 900,
    ):
        self.cache_ttl = cache_ttl_seconds
        # Cache: {feed_url: (timestamp, [entries])}
        self._cache: dict[str, tuple[float, list[dict]]] = {}

        # Load config
        config_path = Path(__file__).parent.parent.parent / feeds_config_path
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self._feeds = config.get("feeds", [])
            self._aliases = config.get("ticker_aliases", {})
        else:
            log.warning("rss_config_not_found", path=str(config_path))
            self._feeds = []
            self._aliases = {}

        log.info("rss_provider_initialized", feeds=len(self._feeds), aliases=len(self._aliases))

    @property
    def provider_name(self) -> str:
        return "rss"

    def search(
        self,
        query: str,
        ticker: str,
        max_results: int = 10,
    ) -> list[dict]:
        """Search RSS feeds for ticker mentions.

        Returns list of dicts with keys: title, url, snippet, source, date.
        """
        self._refresh_stale_feeds()

        # Build search terms: ticker + aliases
        search_terms = [ticker.lower()]
        if ticker in self._aliases:
            search_terms.extend(a.lower() for a in self._aliases[ticker])

        # Also add query words (minus common stop words)
        stop_words = {"stock", "price", "news", "analysis", "market", "the", "and", "for", "in"}
        for word in query.lower().split():
            if word not in stop_words and len(word) > 2:
                search_terms.append(word)

        matches = []
        for feed_url, (_, entries) in self._cache.items():
            for entry in entries:
                if self._entry_matches(entry, search_terms):
                    matches.append(entry)

        # Sort by date (newest first), deduplicate by URL
        matches.sort(key=lambda e: e.get("date", ""), reverse=True)
        seen_urls = set()
        unique = []
        for m in matches:
            if m["url"] not in seen_urls:
                seen_urls.add(m["url"])
                unique.append(m)
            if len(unique) >= max_results:
                break

        log.info(
            "rss_search_complete",
            ticker=ticker,
            terms=len(search_terms),
            matches=len(unique),
            cache_size=sum(len(v[1]) for v in self._cache.values()),
        )
        return unique

    def _refresh_stale_feeds(self) -> int:
        """Fetch feeds where cache has expired. Returns count of refreshed feeds."""
        fp = _get_feedparser()
        now = time.time()
        refreshed = 0

        for feed_meta in self._feeds:
            url = feed_meta["url"]
            cached = self._cache.get(url)
            if cached and (now - cached[0]) < self.cache_ttl:
                continue

            try:
                parsed = fp.parse(url)
                entries = []
                for e in parsed.entries[:50]:  # Cap per feed
                    pub_date = ""
                    if hasattr(e, "published"):
                        pub_date = e.published
                    elif hasattr(e, "updated"):
                        pub_date = e.updated

                    title = getattr(e, "title", "")
                    link = getattr(e, "link", "")
                    summary = getattr(e, "summary", getattr(e, "description", ""))
                    # Strip HTML tags from summary
                    import re

                    summary = re.sub(r"<[^>]+>", "", summary)[:300]

                    entries.append(
                        {
                            "title": title[:150],
                            "url": link,
                            "snippet": summary,
                            "source": feed_meta.get("name", url),
                            "date": pub_date,
                            "category": feed_meta.get("category", "news"),
                        }
                    )

                self._cache[url] = (now, entries)
                refreshed += 1
            except Exception as e:
                log.debug("rss_feed_fetch_failed", url=url, error=str(e))
                # Serve stale cache if available
                if url not in self._cache:
                    self._cache[url] = (now, [])

        if refreshed:
            log.info("rss_feeds_refreshed", count=refreshed, total=len(self._feeds))
        return refreshed

    def _entry_matches(self, entry: dict, search_terms: list[str]) -> bool:
        """Check if entry mentions any search term."""
        text = f"{entry.get('title', '')} {entry.get('snippet', '')}".lower()
        return any(term in text for term in search_terms)
