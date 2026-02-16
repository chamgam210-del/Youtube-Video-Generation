"""Agentic web research pipeline for finding video clips from any source.

Instead of relying solely on YouTube search via SerpAPI, this module uses a
multi-step research approach:

1. **Research Agent** – An LLM analyzes the transcript/query and generates
   diverse, targeted Google search queries to find the best clips.
2. **Google Search** – Uses SerpAPI Google search (web + video) to find
   articles, Reddit threads, social media posts that reference specific clips.
3. **Web Scraper** – Uses Playwright to visit promising pages and extract
   video links from ANY platform (YouTube, Twitter/X, Instagram, TikTok,
   Vimeo, Dailymotion, Facebook, Reddit, Rumble, etc.) with context.
4. **Evaluation Agent** – An LLM scores and ranks all discovered clips based
   on relevance, specificity, and quality signals.
5. **Download & Prepare** – Uses yt-dlp (which supports 1000+ sites) to
   download the winning clip from whatever platform it's hosted on.

This approach finds clips that YouTube search alone cannot surface — e.g.
a specific Megyn Kelly segment on Piers Morgan's show about the Bad Bunny
halftime show, which may be buried on YouTube but linked from news articles,
or a viral Twitter/X clip that was never uploaded to YouTube.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

from .clip_tools import (
    VideoSearchResult,
    PreparedClip,
    download_clip,
    download_clip_section,
    get_video_info,
    find_best_clip_segment,
    get_video_duration,
    trim_clip,
    search_video_clips,
    _parse_view_count,
)


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class DiscoveredClip:
    """A video clip found through web research with rich context.

    Supports clips from any platform that yt-dlp can download:
    YouTube, Twitter/X, Instagram, TikTok, Vimeo, Dailymotion,
    Facebook, Reddit, Rumble, Twitch, and 1000+ more.
    """

    url: str
    title: str
    context: str  # surrounding text from the page where we found this link
    source_page: str  # the web page URL where we found this
    source_type: str  # "google_search", "google_video", "article_scrape", "youtube_search"
    platform: str = ""  # "youtube", "twitter", "tiktok", etc. (auto-detected)
    view_count: int | None = None
    duration_seconds: float | None = None
    relevance_score: float = 0.0  # LLM-assigned 0-1 score


@dataclass
class ResearchResult:
    """Complete output of the research pipeline for one clip query."""

    query: str
    discovered_clips: list[DiscoveredClip] = field(default_factory=list)
    search_queries_used: list[str] = field(default_factory=list)
    pages_scraped: list[str] = field(default_factory=list)
    best_clip: DiscoveredClip | None = None


# ── LLM helpers ──────────────────────────────────────────────────────────────


def _openai_chat(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout_s: int = 60,
    max_retries: int = 5,
) -> str:
    import requests
    import time as _time

    _payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    _headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for _attempt in range(max_retries + 1):
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=_headers,
            json=_payload,
            timeout=timeout_s,
        )
        if resp.status_code == 429 and _attempt < max_retries:
            _wait = min(int(resp.headers.get("Retry-After", 0)) or (30 * (_attempt + 1)), 120)
            print(f"[LLM] 429 rate-limited, retrying in {_wait}s (attempt {_attempt + 1}/{max_retries})")
            _time.sleep(_wait)
            continue
        resp.raise_for_status()
        break
    content = resp.json()["choices"][0]["message"]["content"].strip()
    # Strip markdown fences.
    if content.startswith("```"):
        first_nl = content.index("\n") if "\n" in content else 3
        content = content[first_nl + 1 :]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    return content


# ── Step 1: Research Agent — generate search queries ─────────────────────────


def _generate_research_queries(
    clip_query: str,
    *,
    transcript_context: str = "",
    topic: str = "",
    model: str = "gpt-4o",
) -> list[str]:
    """Use LLM to generate diverse Google search queries to find the right clip.

    Returns 5-8 search queries optimized for Google (not YouTube).
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback: generate basic variations manually.
        return _fallback_search_queries(clip_query, topic)

    system = (
        "You are a research agent tasked with finding a specific video clip on the internet.\n"
        "Given a clip query (describing what we're looking for) and optional context,\n"
        "generate 6-8 diverse Google search queries that would help find this specific clip.\n\n"
        "Strategy:\n"
        "1. Direct video search queries (person + topic + 'clip'/'video')\n"
        "2. News article queries that would LINK to the clip (news sites often embed videos)\n"
        "3. Reddit/social media queries where people share these clips\n"
        "4. Show-specific queries (if the person has a show, search for that show + topic)\n"
        "5. Variation queries with synonyms ('rant', 'reacts', 'goes off', 'slams', 'blasts')\n"
        "6. Platform-specific queries: try 'site:twitter.com', 'site:x.com', 'site:tiktok.com'\n"
        "   in addition to YouTube for clips that may be posted on social media\n\n"
        "IMPORTANT:\n"
        "- Each query should be a Google search query (not a platform-specific search)\n"
        "- Search across ALL platforms — YouTube, Twitter/X, TikTok, Instagram, Vimeo, etc.\n"
        "- Include platform-specific 'site:' queries for viral/social clips\n"
        "- Be SPECIFIC — include the person's name, the exact topic/event\n"
        "- Think about what WEBPAGE would link to this clip\n\n"
        "Return ONLY valid JSON: {\"queries\": [\"query1\", \"query2\", ...]}"
    )

    user_content = {
        "clip_query": clip_query,
        "topic": topic,
        "transcript_context": transcript_context[:1000] if transcript_context else "",
    }

    try:
        content = _openai_chat(
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
            ],
            temperature=0.5,
            max_tokens=1024,
        )
        parsed = json.loads(content)
        queries = parsed.get("queries", [])
        if queries and isinstance(queries, list):
            return [str(q).strip() for q in queries if str(q).strip()][:8]
    except Exception as e:
        print(f"[researcher] LLM query generation failed: {e}")

    return _fallback_search_queries(clip_query, topic)


def _fallback_search_queries(clip_query: str, topic: str = "") -> list[str]:
    """Generate basic search variations without LLM."""
    queries = [clip_query]
    if topic:
        queries.append(f"{clip_query} {topic}")
    queries.extend([
        f"{clip_query} video clip",
        f"{clip_query} video reaction",
        f"{clip_query} reacts rant",
        f"site:youtube.com {clip_query}",
        f"site:twitter.com OR site:x.com {clip_query} video",
    ])
    return queries[:7]


# ── Step 2: Web Search (Playwright Google primary, SerpAPI fallback) ─────────


def _playwright_google_search(
    query: str,
    *,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search Google directly via Playwright (free, unlimited, no API key).

    Launches a headless Chromium, navigates to google.com/search, and parses
    organic results from the DOM.  Much better results than DuckDuckGo.
    """
    try:
        from playwright.sync_api import sync_playwright
        import asyncio as _asyncio

        if sys.platform == "win32":
            _asyncio.set_event_loop_policy(_asyncio.WindowsProactorEventLoopPolicy())

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"],
            )
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 900},
                locale="en-US",
            )
            page = ctx.new_page()
            page.add_init_script(
                'Object.defineProperty(navigator, "webdriver", {get: () => false})'
            )
            url = f"https://www.google.com/search?q={query}&num={max_results}&hl=en"
            page.goto(url, timeout=15000, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)

            # Accept cookie consent if present (common in some locales).
            try:
                consent_btn = page.query_selector("button#L2AGLb")
                if consent_btn:
                    consent_btn.click()
                    page.wait_for_timeout(1500)
            except Exception:
                pass

            results: list[dict[str, Any]] = []
            seen_links: set[str] = set()

            # Parse organic results via <h3> → closest <a> ancestor.
            # Google no longer uses plain div.g in headless; h3-based extraction
            # is more robust.
            for h3 in page.query_selector_all("h3"):
                try:
                    title = (h3.text_content() or "").strip()
                    if not title:
                        continue
                    parent_a = h3.evaluate_handle('el => el.closest("a")')
                    href = parent_a.evaluate("el => el ? el.href : null")
                    if not href or "google.com" in href:
                        continue
                    if href in seen_links:
                        continue
                    seen_links.add(href)
                    # Attempt to extract snippet from a nearby container.
                    snippet = h3.evaluate(
                        """el => {
                            let p = el.parentElement;
                            for (let i = 0; i < 5; i++) {
                                if (p && p.getAttribute("data-hveid")) break;
                                p = p ? p.parentElement : null;
                            }
                            if (!p) p = el.parentElement;
                            let spans = p.querySelectorAll("span");
                            for (let s of spans) {
                                let t = s.textContent.trim();
                                if (t.length > 40 && !t.includes("http")) return t;
                            }
                            return "";
                        }"""
                    )
                    results.append({
                        "title": title[:200],
                        "link": href.strip(),
                        "snippet": (snippet or "")[:300],
                        "source": "playwright_google",
                    })
                    if len(results) >= max_results:
                        break
                except Exception:
                    continue

            browser.close()
            return results

    except ImportError:
        print("[researcher] Playwright not installed — skipping Google search")
        return []
    except Exception as e:
        print(f"[researcher] Playwright Google search failed for {query!r}: {e}")
        return []


def _playwright_google_video_search(
    query: str,
    *,
    max_results: int = 8,
) -> list[dict[str, Any]]:
    """Search Google Videos tab via Playwright (free, unlimited)."""
    try:
        from playwright.sync_api import sync_playwright
        import asyncio as _asyncio

        if sys.platform == "win32":
            _asyncio.set_event_loop_policy(_asyncio.WindowsProactorEventLoopPolicy())

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"],
            )
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 900},
                locale="en-US",
            )
            page = ctx.new_page()
            page.add_init_script(
                'Object.defineProperty(navigator, "webdriver", {get: () => false})'
            )
            # tbm=vid triggers Google Videos tab.
            url = f"https://www.google.com/search?q={query}&tbm=vid&hl=en"
            page.goto(url, timeout=15000, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)

            # Accept cookie consent if present.
            try:
                consent_btn = page.query_selector("button#L2AGLb")
                if consent_btn:
                    consent_btn.click()
                    page.wait_for_timeout(1500)
            except Exception:
                pass

            results: list[dict[str, Any]] = []
            seen_links: set[str] = set()

            # Parse video results via <h3> → closest <a>.
            for h3 in page.query_selector_all("h3"):
                try:
                    title = (h3.text_content() or "").strip()
                    if not title:
                        continue
                    parent_a = h3.evaluate_handle('el => el.closest("a")')
                    href = parent_a.evaluate("el => el ? el.href : null")
                    if not href or href in seen_links:
                        continue
                    # Skip Google's own pages.
                    if "google.com/" in href and "/search" in href:
                        continue
                    seen_links.add(href)
                    results.append({
                        "title": title[:200],
                        "link": href.strip(),
                        "snippet": "",
                        "source": "playwright_google_video",
                        "duration": "",
                    })
                    if len(results) >= max_results:
                        break
                except Exception:
                    continue

            browser.close()
            return results

    except ImportError:
        print("[researcher] Playwright not installed — skipping Google Video search")
        return []
    except Exception as e:
        print(f"[researcher] Playwright Google Video search failed for {query!r}: {e}")
        return []


def _google_search(
    query: str,
    *,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search Google via SerpAPI and return organic results."""

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return []

    import requests

    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": max_results,
        }
        # Retry with backoff on 429 rate-limit errors.
        data = None
        for attempt in range(4):
            resp = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
            if resp.status_code == 429:
                wait = 5 * (2 ** attempt)  # 5, 10, 20, 40s
                print(f"[researcher] SerpAPI 429 — retrying in {wait}s (attempt {attempt+1}/4)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        if data is None:
            return []

        results = []
        for item in data.get("organic_results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", ""),
            })

        # Also grab video results if present.
        for item in data.get("video_results", [])[:5]:
            link = item.get("link", "")
            results.append({
                "title": item.get("title", ""),
                "link": link,
                "snippet": item.get("snippet", ""),
                "source": "video_result",
            })

        return results
    except Exception as e:
        print(f"[researcher] Google search failed for {query!r}: {e}")
        return []


def _google_video_search(
    query: str,
    *,
    max_results: int = 8,
) -> list[dict[str, Any]]:
    """Search Google Videos via SerpAPI — returns video-specific results."""

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return []

    import requests

    try:
        params = {
            "engine": "google_videos",
            "q": query,
            "api_key": api_key,
        }
        # Retry with backoff on 429 rate-limit errors.
        data = None
        for attempt in range(4):
            resp = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
            if resp.status_code == 429:
                wait = 5 * (2 ** attempt)  # 5, 10, 20, 40s
                print(f"[researcher] SerpAPI 429 — retrying in {wait}s (attempt {attempt+1}/4)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        if data is None:
            return []

        results = []
        for item in data.get("video_results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", item.get("description", "")),
                "source": item.get("source", ""),
                "duration": item.get("rich_snippet", {}).get("duration", ""),
            })
        return results
    except Exception as e:
        print(f"[researcher] Google Video search failed for {query!r}: {e}")
        return []


def _web_search(
    query: str,
    *,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Unified web search: Playwright Google → SerpAPI fallback."""
    # 1) Playwright scrapes Google directly (free, unlimited).
    results = _playwright_google_search(query, max_results=max_results)
    if results:
        return results
    # 2) SerpAPI fallback (needs API key + has quota).
    return _google_search(query, max_results=max_results)


def _video_search(
    query: str,
    *,
    max_results: int = 8,
) -> list[dict[str, Any]]:
    """Unified video search: Playwright Google Videos → SerpAPI fallback."""
    results = _playwright_google_video_search(query, max_results=max_results)
    if results:
        return results
    return _google_video_search(query, max_results=max_results)


# ── Step 3: Extract video URLs from web pages (any platform) ─────────────────


# Regex patterns for known video platforms that yt-dlp supports.
# Each maps platform_name → compiled regex.
_YT_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
    re.IGNORECASE,
)

# Master regex matching video URLs from any major platform.
_VIDEO_URL_RE = re.compile(
    r"https?://(?:"
    # YouTube
    r"(?:www\.)?(?:youtube\.com/(?:watch\?[^\s\"'<>]*v=|embed/|shorts/)|youtu\.be/)[a-zA-Z0-9_-]+"
    r"|"
    # Twitter / X
    r"(?:(?:twitter|x)\.com/[a-zA-Z0-9_]+/status/\d+)"
    r"|"
    # TikTok
    r"(?:(?:www\.)?tiktok\.com/@[a-zA-Z0-9_.]+/video/\d+|vm\.tiktok\.com/[a-zA-Z0-9]+)"
    r"|"
    # Instagram (Reels & posts)
    r"(?:(?:www\.)?instagram\.com/(?:reel|p|tv)/[a-zA-Z0-9_-]+)"
    r"|"
    # Facebook / FB Watch
    r"(?:(?:www\.)?facebook\.com/(?:[a-zA-Z0-9.]+/videos/\d+|watch/?\?v=\d+|reel/\d+))"
    r"|"
    # Vimeo
    r"(?:(?:www\.)?vimeo\.com/\d+)"
    r"|"
    # Dailymotion
    r"(?:(?:www\.)?dailymotion\.com/video/[a-zA-Z0-9]+)"
    r"|"
    # Reddit video posts
    r"(?:(?:www\.)?reddit\.com/r/[a-zA-Z0-9_]+/comments/[a-zA-Z0-9]+/[a-zA-Z0-9_]*)"
    r"|"
    # Rumble
    r"(?:(?:www\.)?rumble\.com/v[a-zA-Z0-9]+-[a-zA-Z0-9-]+\.html)"
    r"|"
    # Twitch clips
    r"(?:(?:www\.)?(?:twitch\.tv/[a-zA-Z0-9_]+/clip/|clips\.twitch\.tv/)[a-zA-Z0-9_-]+)"
    r"|"
    # Streamable
    r"(?:(?:www\.)?streamable\.com/[a-zA-Z0-9]+)"
    r"|"
    # BitChute
    r"(?:(?:www\.)?bitchute\.com/video/[a-zA-Z0-9]+)"
    r")",
    re.IGNORECASE,
)

# Domains that host video content (for identifying video links in scraping).
_VIDEO_DOMAINS = {
    "youtube.com", "youtu.be",
    "twitter.com", "x.com",
    "tiktok.com", "vm.tiktok.com",
    "instagram.com",
    "facebook.com", "fb.watch",
    "vimeo.com",
    "dailymotion.com",
    "reddit.com",
    "rumble.com",
    "twitch.tv", "clips.twitch.tv",
    "streamable.com",
    "bitchute.com",
}


def _detect_platform(url: str) -> str:
    """Detect which video platform a URL belongs to."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower().lstrip("www.")
    platform_map = {
        "youtube.com": "youtube",
        "youtu.be": "youtube",
        "twitter.com": "twitter",
        "x.com": "twitter",
        "tiktok.com": "tiktok",
        "vm.tiktok.com": "tiktok",
        "instagram.com": "instagram",
        "facebook.com": "facebook",
        "fb.watch": "facebook",
        "vimeo.com": "vimeo",
        "dailymotion.com": "dailymotion",
        "reddit.com": "reddit",
        "rumble.com": "rumble",
        "twitch.tv": "twitch",
        "clips.twitch.tv": "twitch",
        "streamable.com": "streamable",
        "bitchute.com": "bitchute",
    }
    for key, platform in platform_map.items():
        if domain == key or domain.endswith("." + key):
            return platform
    return ""


def _is_video_url(url: str) -> bool:
    """Check if a URL is from a known video-hosting platform."""
    return bool(_detect_platform(url)) or bool(_VIDEO_URL_RE.match(url))


def _normalize_video_url(url: str) -> str | None:
    """Normalize a video URL.

    For YouTube, normalizes to https://www.youtube.com/watch?v=ID.
    For other platforms, cleans up the URL (strips tracking params, etc.).
    Returns None if URL is not a recognized video URL.
    """
    if not url:
        return None

    # YouTube — canonical normalization.
    m = _YT_URL_RE.search(url)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(1)}"

    # Other platforms — check if it's a video URL and clean it.
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    if _detect_platform(url):
        # Strip common tracking params but keep essential query params.
        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        # Keep query params for platforms that need them (e.g., Facebook ?v=).
        if parsed.query:
            essential_params = {"v", "id", "t", "start"}
            qs = parse_qs(parsed.query)
            kept = {k: v[0] for k, v in qs.items() if k in essential_params}
            if kept:
                clean += "?" + "&".join(f"{k}={v}" for k, v in kept.items())
        return clean

    return None


def _extract_video_urls_from_text(text: str) -> list[str]:
    """Find all video URLs (any platform) in a block of text."""
    found: set[str] = set()
    for m in _VIDEO_URL_RE.finditer(text):
        url = m.group(0)
        norm = _normalize_video_url(url)
        if norm:
            found.add(norm)
    return list(found)


def _scrape_page_for_video_links(
    url: str,
    *,
    timeout_ms: int = 15000,
) -> list[DiscoveredClip]:
    """Use Playwright to load a page and extract video links from ANY platform.

    Looks for:
    - Video embed iframes (YouTube, Twitter, TikTok, Instagram, Vimeo, etc.)
    - Links to video platforms in <a> tags
    - <video> elements with src attributes
    - Open Graph / Twitter Card video meta tags
    - Video URLs in raw page HTML

    Returns DiscoveredClip entries for each video link found.
    """

    clips: list[DiscoveredClip] = []

    # If the URL itself IS a video platform page, return it directly.
    if _is_video_url(url):
        norm = _normalize_video_url(url)
        if norm:
            clips.append(DiscoveredClip(
                url=norm,
                title="",
                context="Direct video URL from search results",
                source_page=url,
                source_type="google_search",
                platform=_detect_platform(norm),
            ))
        return clips

    try:
        from playwright.sync_api import sync_playwright

        # On Windows the default SelectorEventLoop cannot spawn subprocesses;
        # Playwright needs ProactorEventLoop to launch its browser driver.
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 720},
            )
            page = context.new_page()

            try:
                page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                # Wait a bit for dynamic content.
                page.wait_for_timeout(2000)
            except Exception:
                browser.close()
                return clips

            seen_urls_local: set[str] = set()

            def _add_clip(vid_url: str, title: str = "", ctx: str = "", source_type: str = "article_scrape") -> None:
                norm = _normalize_video_url(vid_url)
                if norm and norm not in seen_urls_local:
                    seen_urls_local.add(norm)
                    clips.append(DiscoveredClip(
                        url=norm,
                        title=title[:200],
                        context=ctx[:300],
                        source_page=url,
                        source_type=source_type,
                        platform=_detect_platform(norm),
                    ))

            # Strategy 1: Find ALL video embeds (iframes from any platform).
            try:
                iframes = page.query_selector_all("iframe[src]")
                for iframe in iframes:
                    src = iframe.get_attribute("src") or ""
                    if _is_video_url(src):
                        parent = iframe.evaluate_handle("el => el.parentElement")
                        ctx = parent.evaluate("el => el.textContent || ''") if parent else ""
                        _add_clip(src, ctx=str(ctx).strip())
            except Exception:
                pass

            # Strategy 2: Find links to ANY video platform in <a> tags.
            try:
                all_links = page.query_selector_all("a[href]")
                for link in all_links:
                    href = link.get_attribute("href") or ""
                    if _is_video_url(href):
                        link_text = link.text_content() or ""
                        try:
                            parent_text = link.evaluate(
                                "el => (el.closest('p') || el.closest('div') || el.parentElement)?.textContent || ''"
                            )
                        except Exception:
                            parent_text = ""
                        ctx = f"{link_text} — {parent_text}".strip()
                        _add_clip(href, title=link_text.strip(), ctx=ctx)
            except Exception:
                pass

            # Strategy 3: Find <video> elements with playable sources.
            try:
                videos = page.query_selector_all("video[src], video source[src]")
                for vid in videos:
                    src = vid.get_attribute("src") or ""
                    if src and src.startswith("http"):
                        _add_clip(src, ctx="Embedded <video> element")
            except Exception:
                pass

            # Strategy 4: Check Open Graph / Twitter Card video meta tags.
            try:
                meta_selectors = [
                    'meta[property="og:video"]',
                    'meta[property="og:video:url"]',
                    'meta[property="og:video:secure_url"]',
                    'meta[name="twitter:player"]',
                    'meta[name="twitter:player:stream"]',
                ]
                for sel in meta_selectors:
                    metas = page.query_selector_all(sel)
                    for meta in metas:
                        content = meta.get_attribute("content") or ""
                        if content and content.startswith("http"):
                            _add_clip(content, ctx="Open Graph / Twitter Card video meta tag")
            except Exception:
                pass

            # Strategy 5: Search page HTML for video URLs we might have missed.
            try:
                page_html = page.content()
                for vid_url in _extract_video_urls_from_text(page_html):
                    _add_clip(vid_url, ctx="Found in page HTML")
            except Exception:
                pass

            browser.close()

    except ImportError:
        print("[researcher] Playwright not installed — skipping page scrape")
    except Exception as e:
        print(f"[researcher] Failed to scrape {url}: {e}")

    return clips


# ── Step 4: Evaluation Agent — score and rank clips ──────────────────────────


def _evaluate_clips_with_llm(
    clips: list[DiscoveredClip],
    *,
    original_query: str,
    topic: str = "",
    model: str = "gpt-4o",
) -> list[DiscoveredClip]:
    """Use LLM to score and rank discovered clips by relevance.

    Returns the clips sorted by relevance_score (highest first).
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not clips:
        return clips

    # Deduplicate by URL first.
    seen: set[str] = set()
    unique: list[DiscoveredClip] = []
    for c in clips:
        if c.url not in seen:
            seen.add(c.url)
            unique.append(c)
    clips = unique[:20]  # limit for LLM context

    system = (
        "You are evaluating video clips found through web research across multiple platforms\n"
        "(YouTube, Twitter/X, TikTok, Instagram, Vimeo, Reddit, etc.).\n"
        "For each clip, assign a relevance score from 0.0 to 1.0 based on how well it matches\n"
        "what we're looking for.\n\n"
        "Scoring criteria:\n"
        "- 1.0: PERFECT match — the exact clip we're looking for (right person, right topic, right moment)\n"
        "- 0.8-0.9: Very good — right person and topic, might not be the exact moment\n"
        "- 0.5-0.7: Decent — related to the topic but not the specific clip we want\n"
        "- 0.2-0.4: Weak — tangentially related\n"
        "- 0.0-0.1: Irrelevant\n\n"
        "Key factors:\n"
        "- Does the clip title mention the SPECIFIC PERSON we're looking for?\n"
        "- Is it a REACTION/RESPONSE clip (not a preview or unrelated content)?\n"
        "- Is it from a CREDIBLE source (news outlet, official channel, verified account)?\n"
        "- Does the context from the web page confirm this is the right clip?\n"
        "- Platform quality: YouTube/Vimeo clips are often higher quality; Twitter/TikTok\n"
        "  clips may be shorter but more timely/viral. Prefer the best match regardless of platform.\n\n"
        "Return ONLY valid JSON: {\"scores\": [{\"index\": 0, \"score\": 0.95, \"reason\": \"...\"}, ...]}"
    )

    clip_data = []
    for i, c in enumerate(clips):
        clip_data.append({
            "index": i,
            "url": c.url,
            "title": c.title[:150] if c.title else "(unknown title)",
            "platform": c.platform or _detect_platform(c.url) or "unknown",
            "context": c.context[:200] if c.context else "",
            "source_page": c.source_page[:100],
            "source_type": c.source_type,
        })

    user_content = {
        "looking_for": original_query,
        "topic": topic,
        "clips": clip_data,
    }

    try:
        content = _openai_chat(
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        parsed = json.loads(content)
        scores = parsed.get("scores", [])
        for score_entry in scores:
            idx = int(score_entry.get("index", -1))
            if 0 <= idx < len(clips):
                clips[idx].relevance_score = float(score_entry.get("score", 0.0))
                reason = score_entry.get("reason", "")
                if reason:
                    print(f"  [{clips[idx].relevance_score:.2f}] {clips[idx].title[:60] or clips[idx].url} — {reason}")
    except Exception as e:
        print(f"[researcher] LLM evaluation failed: {e}")

    clips.sort(key=lambda c: c.relevance_score, reverse=True)
    return clips


# ── Step 5: Get video metadata (title etc.) via yt-dlp ───────────────────────


def _enrich_clip_metadata(clip: DiscoveredClip) -> DiscoveredClip:
    """Fetch video title and metadata if missing. Works for any yt-dlp-supported site."""

    if clip.title:
        return clip

    # Use yt-dlp to get the title without downloading.
    try:
        from .clip_tools import _find_ytdlp

        ytdlp = _find_ytdlp()
        if ytdlp:
            import subprocess

            proc = subprocess.run(
                [ytdlp, "--get-title", "--no-playlist", clip.url],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                clip.title = proc.stdout.strip()
    except Exception:
        pass

    return clip


# ── Main orchestrator ────────────────────────────────────────────────────────


def research_clip(
    clip_query: str,
    *,
    topic: str = "",
    transcript_context: str = "",
    model: str = "gpt-4o",
    max_scrape_pages: int = 5,
    seen_urls: set[str] | None = None,
) -> ResearchResult:
    """Run the full agentic research pipeline for a single clip query.

    Steps:
    1. Generate diverse search queries with LLM
    2. Execute Google searches (web + video)
    3. Scrape top pages for YouTube links
    4. Also do a YouTube-specific search as fallback
    5. Evaluate all discovered clips with LLM
    6. Return ranked results

    Parameters
    ----------
    clip_query : str
        What we're looking for (e.g., "Megyn Kelly reacting to Bad Bunny halftime show with Piers Morgan").
    topic : str
        Broader topic context.
    transcript_context : str
        Relevant portion of the transcript for additional context.
    model : str
        LLM model to use for research and evaluation.
    max_scrape_pages : int
        Maximum number of web pages to scrape with Playwright.
    seen_urls : set[str] | None
        URLs to skip (already used by previous clips).
    """

    result = ResearchResult(query=clip_query)

    print(f"\n{'='*70}")
    print(f"[researcher] Starting agentic research for: {clip_query!r}")
    print(f"{'='*70}")

    # ── Step 1: Generate research queries ──
    print("\n[researcher] Step 1: Generating search queries...")
    queries = _generate_research_queries(
        clip_query,
        transcript_context=transcript_context,
        topic=topic,
        model=model,
    )
    result.search_queries_used = queries
    for i, q in enumerate(queries):
        print(f"  Q{i+1}: {q}")

    # ── Step 2: Search the web (Playwright Google primary, SerpAPI fallback) ──
    print("\n[researcher] Step 2: Searching Google...")
    all_search_results: list[dict[str, Any]] = []

    # Use first 3 queries for full web search, rest for video search.
    for q in queries[:3]:
        results = _web_search(q, max_results=8)
        all_search_results.extend(results)
        print(f"  Web search: {len(results)} results for {q!r}")
        time.sleep(2)  # polite delay between searches

    for q in queries[:3]:
        results = _video_search(q, max_results=6)
        all_search_results.extend(results)
        print(f"  Video search: {len(results)} results for {q!r}")
        time.sleep(2)

    # ── Step 3: Extract video URLs from search results + scrape pages ──
    print("\n[researcher] Step 3: Extracting video URLs and scraping pages...")

    # Direct video URLs from search results (any platform).
    for sr in all_search_results:
        link = sr.get("link", "")
        norm = _normalize_video_url(link)
        if norm:
            result.discovered_clips.append(DiscoveredClip(
                url=norm,
                title=sr.get("title", ""),
                context=sr.get("snippet", ""),
                source_page=link,
                source_type="google_search",
                platform=_detect_platform(norm),
            ))

    # Identify non-video pages worth scraping (news articles, Reddit, etc.).
    pages_to_scrape: list[str] = []
    scraped_domains: set[str] = set()
    for sr in all_search_results:
        link = sr.get("link", "")
        if not link:
            continue
        parsed = urlparse(link)
        domain = parsed.netloc.lower()
        # Skip pages that are themselves video platform pages (already handled above).
        if _is_video_url(link):
            continue
        # Skip domains we've already scraped.
        if domain in scraped_domains:
            continue
        # Prioritize news sites, Reddit, and trusted sources.
        priority_domains = {
            "reddit.com", "foxnews.com", "cnn.com", "msnbc.com",
            "nbcnews.com", "bbc.com", "theguardian.com", "nytimes.com",
            "washingtonpost.com", "dailywire.com", "mediaite.com",
            "thehill.com", "politico.com", "breitbart.com", "huffpost.com",
            "variety.com", "deadline.com", "hollywoodreporter.com",
            "ew.com", "people.com", "tmz.com", "buzzfeed.com",
            "thedailybeast.com", "salon.com", "vox.com",
        }
        is_priority = any(pd in domain for pd in priority_domains)
        if is_priority and len(pages_to_scrape) < max_scrape_pages:
            pages_to_scrape.append(link)
            scraped_domains.add(domain)

    # If we don't have enough priority pages, add any non-video-platform page.
    for sr in all_search_results:
        if len(pages_to_scrape) >= max_scrape_pages:
            break
        link = sr.get("link", "")
        if not link:
            continue
        parsed = urlparse(link)
        domain = parsed.netloc.lower()
        if _is_video_url(link):
            continue
        if domain in scraped_domains:
            continue
        pages_to_scrape.append(link)
        scraped_domains.add(domain)

    # Scrape pages for embedded video links (any platform).
    for page_url in pages_to_scrape:
        print(f"  Scraping: {page_url[:80]}...")
        scraped = _scrape_page_for_video_links(page_url)
        result.discovered_clips.extend(scraped)
        result.pages_scraped.append(page_url)
        platforms_found = set(c.platform for c in scraped if c.platform)
        print(f"    Found {len(scraped)} video links ({', '.join(platforms_found) or 'none'})")

    # ── Step 3b: Also do a direct YouTube search as fallback ──
    print("\n[researcher] Step 3b: YouTube search fallback...")
    # Enrich short queries with topic for better YouTube search results.
    import re as _re
    _yt_query = _re.sub(r'\bNone\b', '', clip_query).strip()
    if topic and len(_yt_query.split()) <= 4 and topic.lower() not in _yt_query.lower():
        _yt_query = f"{_yt_query} {topic}"
    yt_results = search_video_clips(_yt_query, max_results=8, sort_by_views=False)
    for r in yt_results:
        result.discovered_clips.append(DiscoveredClip(
            url=r.url,
            title=r.title,
            context=f"YouTube search result ({r.view_count or 0:,} views)",
            source_page="youtube.com",
            source_type="youtube_search",
            platform="youtube",
            view_count=r.view_count,
            duration_seconds=r.duration_seconds,
        ))

    # Deduplicate by normalized URL (works across all platforms).
    seen: set[str] = set(seen_urls or set())
    unique: list[DiscoveredClip] = []
    for c in result.discovered_clips:
        norm = _normalize_video_url(c.url)
        key = norm or c.url  # fall back to raw URL if normalization fails
        if key not in seen:
            seen.add(key)
            if norm:
                c.url = norm
            if not c.platform:
                c.platform = _detect_platform(c.url)
            unique.append(c)
    result.discovered_clips = unique

    print(f"\n[researcher] Total unique clips discovered: {len(result.discovered_clips)}")

    # ── Step 4: Enrich metadata for clips without titles ──
    print("\n[researcher] Step 4: Enriching clip metadata...")
    for c in result.discovered_clips:
        if not c.title:
            _enrich_clip_metadata(c)
            if c.title:
                print(f"  Enriched: {c.url} → {c.title[:60]}")

    # ── Step 5: Evaluate and rank clips with LLM ──
    print("\n[researcher] Step 5: Evaluating clips with LLM...")
    result.discovered_clips = _evaluate_clips_with_llm(
        result.discovered_clips,
        original_query=clip_query,
        topic=topic,
        model=model,
    )

    if result.discovered_clips:
        result.best_clip = result.discovered_clips[0]
        print(f"\n[researcher] Best clip: {result.best_clip.title[:80]} ({result.best_clip.relevance_score:.2f})")
        print(f"  URL: {result.best_clip.url}")
    else:
        print("\n[researcher] No clips found!")

    return result


# ── High-level: research → download → prepare ───────────────────────────────


def _research_and_download_one(
    search_query: str,
    *,
    target_duration: float,
    dest_dir: Path,
    raw_dir: Path,
    clip_index: int,
    width: int,
    height: int,
    topic: str,
    transcript_context: str,
    llm_model: str,
    seen_urls: set[str] | None,
    max_scrape_pages: int,
    mute: bool,
    timeline_start: float,
    timeline_end: float,
) -> PreparedClip | None:
    """Research a single query, download the best matching clip, return PreparedClip or None."""

    research = research_clip(
        search_query,
        topic=topic,
        transcript_context=transcript_context,
        model=llm_model,
        max_scrape_pages=max_scrape_pages,
        seen_urls=seen_urls,
    )

    if not research.best_clip:
        return None

    for rank, clip in enumerate(research.discovered_clips[:5]):
        if clip.relevance_score < 0.3 and rank > 0:
            break
        print(f"[researcher] Trying clip #{rank+1}: {clip.title[:60] or clip.url}")

        info = get_video_info(clip.url)
        video_duration = 0.0
        video_title = clip.title or ""
        if info:
            video_duration = info.get("duration", 0.0)
            if not video_title:
                video_title = info.get("title", "")
            print(f"  Duration: {video_duration:.0f}s, Title: {video_title[:60]}")
        else:
            print(f"  Could not get video info, trying download anyway")

        clip_start = 0.0
        clip_dur = target_duration
        if video_duration > 0:
            import os as _os, json as _json
            api_key = _os.getenv("OPENAI_API_KEY")
            if api_key and video_duration > target_duration * 1.5:
                try:
                    import requests as _req
                    system = (
                        "You are helping select the best segment from a video to use as a clip.\n"
                        "Given the video title, total duration, what we searched for, and how long the clip should be,\n"
                        "pick the best START time (in seconds) so the clip shows the most relevant/interesting part.\n\n"
                        "Guidelines:\n"
                        "- Skip intros, outros, channel branding (usually first 5-15s and last 10s).\n"
                        "- For news clips: jump to where the person of interest is actually speaking.\n"
                        "- For reaction videos: jump to the peak reaction moment.\n"
                        "- For interviews: jump to the key quote or heated exchange.\n"
                        "- For podcast episodes / long shows: estimate where in the episode the topic\n"
                        "  would be discussed (often 10-30% in, after intro/ads).\n"
                        "- For social media clips (Twitter, TikTok): they're usually short — start near 0.\n"
                        "- Make sure start + duration doesn't exceed the total video duration.\n\n"
                        'Return ONLY valid JSON: {"start": <float>}'
                    )
                    user_msg = {
                        "video_title": video_title,
                        "video_duration_seconds": round(video_duration, 1),
                        "search_query": search_query,
                        "desired_clip_duration": round(target_duration, 1),
                    }
                    resp = _req.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": llm_model, "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": _json.dumps(user_msg, ensure_ascii=False)},
                        ], "temperature": 0.2, "max_tokens": 60},
                        timeout=20,
                    )
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"].strip()
                    if content.startswith("```"):
                        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                    parsed = _json.loads(content)
                    clip_start = float(parsed.get("start", 0))
                    clip_start = max(0.0, min(clip_start, video_duration - target_duration))
                    clip_dur = min(target_duration, video_duration - clip_start)
                    print(f"  LLM picked segment: {clip_start:.1f}s - {clip_start + clip_dur:.1f}s")
                except Exception as e:
                    print(f"  LLM segment selection failed: {e}")
                    clip_start = max(0.0, video_duration * 0.15)
                    clip_dur = min(target_duration, video_duration - clip_start)
            else:
                clip_start = 0.0
                clip_dur = min(target_duration, video_duration)
        else:
            clip_start = 0.0
            clip_dur = target_duration

        section_end = clip_start + clip_dur
        raw_path = download_clip_section(
            clip.url, raw_dir,
            start=clip_start, end=section_end,
            prefix=f"clip{clip_index:02d}",
        )
        if raw_path is None:
            print(f"  Download failed, trying next...")
            continue

        if seen_urls is not None:
            seen_urls.add(clip.url)

        actual_dur = get_video_duration(raw_path)
        trim_start = 5.0 if actual_dur > clip_dur + 3.0 else 0.0
        trim_dur = min(clip_dur, actual_dur - trim_start) if actual_dur > 0 else clip_dur

        trimmed_path = dest_dir / f"prepared_{clip_index:02d}.mp4"
        try:
            trim_clip(raw_path, trimmed_path, start=trim_start, duration=trim_dur,
                      width=width, height=height, mute=mute)
        except Exception as e:
            print(f"[researcher] Trim failed: {e}")
            continue

        return PreparedClip(
            path=trimmed_path,
            timeline_start=timeline_start,
            timeline_end=timeline_end,
            source_url=clip.url,
            source_title=clip.title or "",
            search_query=search_query,
            muted=mute,
        )

    return None


def research_and_prepare_clip(
    suggestion: Any,  # VideoClipSuggestion or CommentaryClipSuggestion
    *,
    dest_dir: str | Path,
    clip_index: int = 0,
    width: int = 1920,
    height: int = 1080,
    topic: str = "",
    transcript_context: str = "",
    llm_model: str = "gpt-4o",
    seen_urls: set[str] | None = None,
    max_scrape_pages: int = 5,
    max_clips: int = 1,
) -> list[PreparedClip]:
    """Full pipeline: research → download → trim → return prepared clips.

    This replaces ``prepare_clip_for_suggestion`` with an agentic research approach.
    Returns a list of PreparedClip (possibly empty).

    For **compilation** suggestions with *extra_queries*, each query is researched
    independently so the resulting clips come from different people/sources.
    For **reference** suggestions (or compilations without extra_queries), the
    top *max_clips* results from a single research run are downloaded.
    """

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    target_duration = float(suggestion.timeline_end - suggestion.timeline_start)
    if target_duration <= 0:
        return []

    raw_dir = dest_dir / "raw"
    mute = getattr(suggestion, "mute", False)
    extra_queries: list[str] = getattr(suggestion, "extra_queries", None) or []
    clip_type = getattr(suggestion, "clip_type", "reference")

    common_kw = dict(
        target_duration=target_duration / max(1, max_clips),  # per-clip duration
        dest_dir=dest_dir,
        raw_dir=raw_dir,
        width=width,
        height=height,
        topic=topic,
        transcript_context=transcript_context,
        llm_model=llm_model,
        seen_urls=seen_urls,
        max_scrape_pages=max_scrape_pages,
        mute=mute,
        timeline_start=suggestion.timeline_start,
        timeline_end=suggestion.timeline_end,
    )

    results: list[PreparedClip] = []

    # ── Compilation with extra_queries: research each query independently ──
    if clip_type == "compilation" and extra_queries:
        # Build the full query list: primary + extras.
        all_queries = [suggestion.search_query] + list(extra_queries)
        # Limit to max_clips.
        all_queries = all_queries[:max_clips]

        # Sanitize: strip literal "None" that can leak from LLM JSON,
        # and enrich short queries (just a name) with topic context.
        import re as _re
        _clean_queries: list[str] = []
        for _rq in all_queries:
            _rq = _re.sub(r'\bNone\b', '', str(_rq)).strip()
            # If query is very short (just a name), append topic for better results.
            if topic and len(_rq.split()) <= 4 and topic.lower() not in _rq.lower():
                _rq = f"{_rq} {topic}"
            if _rq:
                _clean_queries.append(_rq)
        all_queries = _clean_queries

        print(f"[researcher] Compilation: researching {len(all_queries)} separate queries")
        for qi, q in enumerate(all_queries):
            idx = clip_index + len(results)
            print(f"\n[researcher] === Compilation clip {qi+1}/{len(all_queries)}: {q} ===")
            pc = _research_and_download_one(q, clip_index=idx, **common_kw)
            if pc is not None:
                results.append(pc)
                print(f"  ✓ Clip {len(results)}/{len(all_queries)} prepared")
            else:
                print(f"  ✗ No clip found for: {q}")
        return results

    # ── Single query (reference) or compilation without extra_queries ──
    # Run one research, download up to max_clips from the results.
    research = research_clip(
        suggestion.search_query,
        topic=topic,
        transcript_context=transcript_context,
        model=llm_model,
        max_scrape_pages=max_scrape_pages,
        seen_urls=seen_urls,
    )

    if not research.best_clip:
        return []

    clips_downloaded = 0
    for rank, clip in enumerate(research.discovered_clips[:5 + max_clips]):
        if clips_downloaded >= max_clips:
            break
        if clip.relevance_score < 0.3 and rank > 0:
            break
        idx = clip_index + clips_downloaded
        print(f"[researcher] Trying clip #{rank+1}: {clip.title[:60] or clip.url}")

        info = get_video_info(clip.url)
        video_duration = 0.0
        video_title = clip.title or ""
        if info:
            video_duration = info.get("duration", 0.0)
            if not video_title:
                video_title = info.get("title", "")
            print(f"  Duration: {video_duration:.0f}s, Title: {video_title[:60]}")
        else:
            print(f"  Could not get video info, trying download anyway")

        per_clip_dur = target_duration / max(1, max_clips)

        clip_start = 0.0
        clip_dur = per_clip_dur
        if video_duration > 0:
            import os as _os, json as _json
            api_key = _os.getenv("OPENAI_API_KEY")
            if api_key and video_duration > per_clip_dur * 1.5:
                try:
                    import requests as _req
                    system = (
                        "You are helping select the best segment from a video to use as a clip.\n"
                        "Given the video title, total duration, what we searched for, and how long the clip should be,\n"
                        "pick the best START time (in seconds) so the clip shows the most relevant/interesting part.\n\n"
                        "Guidelines:\n"
                        "- Skip intros, outros, channel branding (usually first 5-15s and last 10s).\n"
                        "- For news clips: jump to where the person of interest is actually speaking.\n"
                        "- For reaction videos: jump to the peak reaction moment.\n"
                        "- For interviews: jump to the key quote or heated exchange.\n"
                        "- For podcast episodes / long shows: estimate where in the episode the topic\n"
                        "  would be discussed (often 10-30% in, after intro/ads).\n"
                        "- For social media clips (Twitter, TikTok): they're usually short — start near 0.\n"
                        "- Make sure start + duration doesn't exceed the total video duration.\n\n"
                        'Return ONLY valid JSON: {"start": <float>}'
                    )
                    user_msg = {
                        "video_title": video_title,
                        "video_duration_seconds": round(video_duration, 1),
                        "search_query": suggestion.search_query,
                        "desired_clip_duration": round(per_clip_dur, 1),
                    }
                    resp = _req.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": llm_model, "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": _json.dumps(user_msg, ensure_ascii=False)},
                        ], "temperature": 0.2, "max_tokens": 60},
                        timeout=20,
                    )
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"].strip()
                    if content.startswith("```"):
                        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                    parsed = _json.loads(content)
                    clip_start = float(parsed.get("start", 0))
                    clip_start = max(0.0, min(clip_start, video_duration - per_clip_dur))
                    clip_dur = min(per_clip_dur, video_duration - clip_start)
                    print(f"  LLM picked segment: {clip_start:.1f}s - {clip_start + clip_dur:.1f}s")
                except Exception as e:
                    print(f"  LLM segment selection failed: {e}")
                    clip_start = max(0.0, video_duration * 0.15)
                    clip_dur = min(per_clip_dur, video_duration - clip_start)
            else:
                clip_start = 0.0
                clip_dur = min(per_clip_dur, video_duration)
        else:
            clip_start = 0.0
            clip_dur = per_clip_dur

        section_end = clip_start + clip_dur
        raw_path = download_clip_section(
            clip.url, raw_dir,
            start=clip_start, end=section_end,
            prefix=f"clip{idx:02d}",
        )
        if raw_path is None:
            print(f"  Download failed, trying next...")
            continue

        if seen_urls is not None:
            seen_urls.add(clip.url)

        actual_dur = get_video_duration(raw_path)
        trim_start = 5.0 if actual_dur > clip_dur + 3.0 else 0.0
        trim_dur = min(clip_dur, actual_dur - trim_start) if actual_dur > 0 else clip_dur

        trimmed_path = dest_dir / f"prepared_{idx:02d}.mp4"
        try:
            trim_clip(raw_path, trimmed_path, start=trim_start, duration=trim_dur,
                      width=width, height=height, mute=mute)
        except Exception as e:
            print(f"[researcher] Trim failed: {e}")
            continue

        results.append(PreparedClip(
            path=trimmed_path,
            timeline_start=suggestion.timeline_start,
            timeline_end=suggestion.timeline_end,
            source_url=clip.url,
            source_title=clip.title or "",
            search_query=suggestion.search_query,
            muted=mute,
        ))
        clips_downloaded += 1
        print(f"  ✓ Clip {clips_downloaded}/{max_clips} prepared: {trimmed_path.name}")

    return results
