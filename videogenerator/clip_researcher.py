"""Agentic web research pipeline for finding video clips.

Instead of relying solely on YouTube search via SerpAPI, this module uses a
multi-step research approach:

1. **Research Agent** – An LLM analyzes the transcript/query and generates
   diverse, targeted Google search queries to find the best clips.
2. **Google Search** – Uses SerpAPI Google search (web + video) to find
   articles, Reddit threads, social media posts that reference specific clips.
3. **Web Scraper** – Uses Playwright to visit promising pages and extract
   YouTube links with surrounding context (titles, descriptions).
4. **Evaluation Agent** – An LLM scores and ranks all discovered clips based
   on relevance, specificity, and quality signals.
5. **Download & Prepare** – Uses the existing yt-dlp pipeline for the winner.

This approach finds clips that YouTube search alone cannot surface — e.g.
a specific Megyn Kelly segment on Piers Morgan's show about the Bad Bunny
halftime show, which may be buried on YouTube but linked from news articles.
"""

from __future__ import annotations

import json
import os
import re
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
    """A YouTube clip found through web research with rich context."""

    url: str
    title: str
    context: str  # surrounding text from the page where we found this link
    source_page: str  # the web page URL where we found this
    source_type: str  # "google_search", "google_video", "article_scrape", "youtube_search"
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
) -> str:
    import requests

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
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
        "1. Direct YouTube search queries (person + topic + 'clip'/'video')\n"
        "2. News article queries that would LINK to the clip (news sites often embed YouTube videos)\n"
        "3. Reddit/social media queries where people share these clips\n"
        "4. Show-specific queries (if the person has a show, search for that show + topic)\n"
        "5. Variation queries with synonyms ('rant', 'reacts', 'goes off', 'slams', 'blasts')\n\n"
        "IMPORTANT:\n"
        "- Each query should be a Google search query (not a YouTube search)\n"
        "- Include 'youtube' in some queries to find pages linking to YouTube videos\n"
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
        f"{clip_query} youtube clip",
        f"{clip_query} video reaction",
        f"{clip_query} reacts rant",
        f"site:youtube.com {clip_query}",
    ])
    return queries[:6]


# ── Step 2: Google Web Search via SerpAPI ────────────────────────────────────


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
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

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
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

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


# ── Step 3: Extract YouTube URLs from web pages ─────────────────────────────


_YT_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
    re.IGNORECASE,
)


def _normalize_youtube_url(url: str) -> str | None:
    """Normalize any YouTube URL variant to https://www.youtube.com/watch?v=ID."""
    m = _YT_URL_RE.search(url)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(1)}"
    return None


def _extract_youtube_urls_from_text(text: str) -> list[str]:
    """Find all YouTube URLs in a block of text."""
    found = set()
    for m in _YT_URL_RE.finditer(text):
        url = f"https://www.youtube.com/watch?v={m.group(1)}"
        found.add(url)
    return list(found)


def _scrape_page_for_youtube_links(
    url: str,
    *,
    timeout_ms: int = 15000,
) -> list[DiscoveredClip]:
    """Use Playwright to load a page and extract YouTube links with context.

    Returns DiscoveredClip entries for each YouTube link found.
    """

    clips: list[DiscoveredClip] = []

    # Skip YouTube pages themselves (we handle those differently).
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
        norm = _normalize_youtube_url(url)
        if norm:
            clips.append(DiscoveredClip(
                url=norm,
                title="",
                context="Direct YouTube URL from search results",
                source_page=url,
                source_type="google_search",
            ))
        return clips

    try:
        from playwright.sync_api import sync_playwright

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

            # Strategy 1: Find YouTube embeds (iframes).
            try:
                iframes = page.query_selector_all("iframe[src*='youtube.com'], iframe[src*='youtu.be']")
                for iframe in iframes:
                    src = iframe.get_attribute("src") or ""
                    norm = _normalize_youtube_url(src)
                    if norm:
                        # Get surrounding text for context.
                        parent = iframe.evaluate_handle("el => el.parentElement")
                        ctx = parent.evaluate("el => el.textContent || ''") if parent else ""
                        clips.append(DiscoveredClip(
                            url=norm,
                            title="",
                            context=str(ctx).strip()[:300],
                            source_page=url,
                            source_type="article_scrape",
                        ))
            except Exception:
                pass

            # Strategy 2: Find YouTube links in <a> tags.
            try:
                links = page.query_selector_all("a[href*='youtube.com'], a[href*='youtu.be']")
                for link in links:
                    href = link.get_attribute("href") or ""
                    norm = _normalize_youtube_url(href)
                    if norm:
                        link_text = link.text_content() or ""
                        # Get parent paragraph or container for context.
                        try:
                            parent_text = link.evaluate(
                                "el => (el.closest('p') || el.closest('div') || el.parentElement)?.textContent || ''"
                            )
                        except Exception:
                            parent_text = ""
                        ctx = f"{link_text} — {parent_text}".strip()[:300]
                        clips.append(DiscoveredClip(
                            url=norm,
                            title=link_text.strip()[:200],
                            context=ctx,
                            source_page=url,
                            source_type="article_scrape",
                        ))
            except Exception:
                pass

            # Strategy 3: Search page text for YouTube URLs.
            try:
                body_text = page.evaluate("document.body?.innerText || ''")
                page_html = page.content()
                # Find URLs in raw HTML that we might have missed.
                for yt_url in _extract_youtube_urls_from_text(page_html):
                    norm = _normalize_youtube_url(yt_url)
                    if norm and not any(c.url == norm for c in clips):
                        clips.append(DiscoveredClip(
                            url=norm,
                            title="",
                            context="Found in page HTML",
                            source_page=url,
                            source_type="article_scrape",
                        ))
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
        "You are evaluating YouTube video clips found through web research.\n"
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
        "- Is it from a CREDIBLE source (news outlet, official channel)?\n"
        "- Does the context from the web page confirm this is the right clip?\n\n"
        "Return ONLY valid JSON: {\"scores\": [{\"index\": 0, \"score\": 0.95, \"reason\": \"...\"}, ...]}"
    )

    clip_data = []
    for i, c in enumerate(clips):
        clip_data.append({
            "index": i,
            "url": c.url,
            "title": c.title[:150] if c.title else "(unknown title)",
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


# ── Step 5: Get YouTube video metadata (title etc.) ──────────────────────────


def _enrich_clip_metadata(clip: DiscoveredClip) -> DiscoveredClip:
    """Fetch YouTube video title and metadata if missing."""

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

    # ── Step 2: Execute Google searches ──
    print("\n[researcher] Step 2: Searching Google...")
    all_search_results: list[dict[str, Any]] = []

    # Use first 3 queries for full Google search, rest for video search.
    for q in queries[:3]:
        results = _google_search(q, max_results=8)
        all_search_results.extend(results)
        print(f"  Google web: {len(results)} results for {q!r}")
        time.sleep(0.5)  # rate limiting

    for q in queries[:3]:
        results = _google_video_search(q, max_results=6)
        all_search_results.extend(results)
        print(f"  Google video: {len(results)} results for {q!r}")
        time.sleep(0.5)

    # ── Step 3: Extract YouTube URLs from search results + scrape pages ──
    print("\n[researcher] Step 3: Extracting YouTube URLs and scraping pages...")

    # Direct YouTube URLs from search results.
    for sr in all_search_results:
        link = sr.get("link", "")
        norm = _normalize_youtube_url(link)
        if norm:
            result.discovered_clips.append(DiscoveredClip(
                url=norm,
                title=sr.get("title", ""),
                context=sr.get("snippet", ""),
                source_page=link,
                source_type="google_search",
            ))

    # Identify non-YouTube pages worth scraping (news articles, Reddit, etc.).
    pages_to_scrape: list[str] = []
    scraped_domains: set[str] = set()
    for sr in all_search_results:
        link = sr.get("link", "")
        if not link:
            continue
        parsed = urlparse(link)
        domain = parsed.netloc.lower()
        # Skip YouTube (we already extracted those).
        if "youtube.com" in domain or "youtu.be" in domain:
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

    # If we don't have enough priority pages, add any non-YouTube page.
    for sr in all_search_results:
        if len(pages_to_scrape) >= max_scrape_pages:
            break
        link = sr.get("link", "")
        if not link:
            continue
        parsed = urlparse(link)
        domain = parsed.netloc.lower()
        if "youtube.com" in domain or "youtu.be" in domain:
            continue
        if domain in scraped_domains:
            continue
        pages_to_scrape.append(link)
        scraped_domains.add(domain)

    # Scrape pages for embedded YouTube links.
    for page_url in pages_to_scrape:
        print(f"  Scraping: {page_url[:80]}...")
        scraped = _scrape_page_for_youtube_links(page_url)
        result.discovered_clips.extend(scraped)
        result.pages_scraped.append(page_url)
        print(f"    Found {len(scraped)} YouTube links")

    # ── Step 3b: Also do a direct YouTube search as fallback ──
    print("\n[researcher] Step 3b: YouTube search fallback...")
    yt_results = search_video_clips(clip_query, max_results=8, sort_by_views=False)
    for r in yt_results:
        result.discovered_clips.append(DiscoveredClip(
            url=r.url,
            title=r.title,
            context=f"YouTube search result ({r.view_count or 0:,} views)",
            source_page="youtube.com",
            source_type="youtube_search",
            view_count=r.view_count,
            duration_seconds=r.duration_seconds,
        ))

    # Deduplicate by URL.
    seen: set[str] = set(seen_urls or set())
    unique: list[DiscoveredClip] = []
    for c in result.discovered_clips:
        norm = _normalize_youtube_url(c.url)
        if norm and norm not in seen:
            seen.add(norm)
            c.url = norm
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
) -> PreparedClip | None:
    """Full pipeline: research → download → trim → return prepared clip.

    This replaces ``prepare_clip_for_suggestion`` with an agentic research approach.
    Returns None if any step fails (non-fatal).
    """

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    target_duration = float(suggestion.timeline_end - suggestion.timeline_start)
    if target_duration <= 0:
        return None

    # Run agentic research.
    research = research_clip(
        suggestion.search_query,
        topic=topic,
        transcript_context=transcript_context,
        model=llm_model,
        max_scrape_pages=max_scrape_pages,
        seen_urls=seen_urls,
    )

    if not research.best_clip:
        return None

    # Try the best clips in order: get info → pick segment → download only that section.
    raw_dir = dest_dir / "raw"
    raw_path = None
    chosen = None
    clip_start = 0.0
    clip_dur = target_duration

    for rank, clip in enumerate(research.discovered_clips[:5]):
        if clip.relevance_score < 0.3 and rank > 0:
            break  # don't try low-relevance clips
        print(f"[researcher] Trying clip #{rank+1}: {clip.title[:60] or clip.url}")

        # Step A: Get video info (duration, title) without downloading.
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

        # Step B: Use LLM to pick the best segment within the video.
        if video_duration > 0:
            from .clip_tools import find_best_clip_segment as _find_seg
            # We don't have the file yet — use the LLM with title/duration only.
            import os as _os, json as _json
            api_key = _os.getenv("OPENAI_API_KEY")
            if api_key and video_duration > target_duration * 1.5:
                try:
                    import requests as _req
                    system = (
                        "You are helping select the best segment from a YouTube video to use as a clip.\n"
                        "Given the video title, total duration, what we searched for, and how long the clip should be,\n"
                        "pick the best START time (in seconds) so the clip shows the most relevant/interesting part.\n\n"
                        "Guidelines:\n"
                        "- Skip intros, outros, channel branding (usually first 5-15s and last 10s).\n"
                        "- For news clips: jump to where the person of interest is actually speaking.\n"
                        "- For reaction videos: jump to the peak reaction moment.\n"
                        "- For interviews: jump to the key quote or heated exchange.\n"
                        "- For podcast episodes / long shows: estimate where in the episode the topic\n"
                        "  would be discussed (often 10-30% in, after intro/ads).\n"
                        "- Make sure start + duration doesn't exceed the total video duration.\n\n"
                        'Return ONLY valid JSON: {"start": <float>}'
                    )
                    user_msg = {
                        "video_title": video_title,
                        "video_duration_seconds": round(video_duration, 1),
                        "search_query": suggestion.search_query,
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

        # Step C: Download only the relevant section.
        section_end = clip_start + clip_dur
        raw_path = download_clip_section(
            clip.url,
            raw_dir,
            start=clip_start,
            end=section_end,
            prefix=f"clip{clip_index:02d}",
        )
        if raw_path is not None:
            chosen = clip
            if seen_urls is not None:
                seen_urls.add(clip.url)
            break
        print(f"  Download failed, trying next...")

    if raw_path is None or chosen is None:
        return None

    # The downloaded section is already roughly the right segment.
    # Do a final trim + scale to exact dimensions.
    mute = getattr(suggestion, "mute", False)
    actual_dur = get_video_duration(raw_path)
    # The section download has ~5s padding on each side, so trim to center.
    trim_start = 5.0 if actual_dur > clip_dur + 3.0 else 0.0
    trim_dur = min(clip_dur, actual_dur - trim_start) if actual_dur > 0 else clip_dur

    trimmed_path = dest_dir / f"prepared_{clip_index:02d}.mp4"
    try:
        trim_clip(
            raw_path,
            trimmed_path,
            start=trim_start,
            duration=trim_dur,
            width=width,
            height=height,
            mute=mute,
        )
    except Exception as e:
        print(f"[researcher] Trim failed: {e}")
        return None

    return PreparedClip(
        path=trimmed_path,
        timeline_start=suggestion.timeline_start,
        timeline_end=suggestion.timeline_end,
        source_url=chosen.url,
        source_title=chosen.title or "",
        search_query=suggestion.search_query,
        muted=mute,
    )
