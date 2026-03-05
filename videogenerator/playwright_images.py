"""Playwright-based Google Image Search — no API key required.

Uses headless Chromium to search Google Images and extract image URLs
directly from the page DOM.  Returns the same dict format that SerpAPI
and DuckDuckGo providers return so it's a drop-in replacement.

Falls back to DuckDuckGo image search if Playwright is unavailable.
"""

from __future__ import annotations

import sys
from typing import Any
from urllib.parse import quote_plus


def playwright_google_image_search(
    query: str,
    *,
    max_results: int = 20,
    min_width: int = 800,
) -> list[dict[str, Any]]:
    """Search Google Images via Playwright and return image candidates.

    Each result dict contains:
      - title, page_url, image_url, original_url, width, height,
        license_name, license_url, attribution
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
                viewport={"width": 1400, "height": 900},
                locale="en-US",
            )
            page = ctx.new_page()
            page.add_init_script(
                'Object.defineProperty(navigator, "webdriver", {get: () => false})'
            )

            # Add negative keywords to suppress posters/covers in results.
            _search_q = f"{query} -poster -cover -logo -dvd"
            encoded_q = quote_plus(_search_q)
            # tbs=itp:photo = filter to photographs only (no clipart/drawings/posters).
            url = f"https://www.google.com/search?q={encoded_q}&tbm=isch&hl=en&tbs=itp:photo"
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)

            # Accept cookie consent if present.
            try:
                consent_btn = page.query_selector("button#L2AGLb")
                if consent_btn:
                    consent_btn.click()
                    page.wait_for_timeout(1500)
            except Exception:
                pass

            # ── Strategy 1: Extract full-res URLs from page source ──
            # Google embeds original image URLs in the page JS/data attrs.
            results: list[dict[str, Any]] = []
            seen_urls: set[str] = set()

            import re as _re
            import json as _json

            page_content = page.content()

            # Google Images embeds original URLs in multiple formats.
            # Look for patterns like: ["https://...jpg",width,height]
            # or in metadata arrays.
            _IMG_URL_PATTERNS = [
                # Pattern 1: URLs in JS arrays — ["https://...jpg",4000,3000]
                _re.compile(
                    r'\["(https?://[^"]+\.(?:jpg|jpeg|png|webp)(?:\?[^"]*)?)"'
                    r',\s*(\d+)\s*,\s*(\d+)\s*\]',
                    _re.IGNORECASE,
                ),
                # Pattern 2: ou/tu style metadata
                _re.compile(
                    r'"ou"\s*:\s*"(https?://[^"]+)".*?"ow"\s*:\s*(\d+).*?"oh"\s*:\s*(\d+)',
                    _re.IGNORECASE,
                ),
            ]

            # Also try extracting data-src/src from image elements.
            img_elements = page.query_selector_all("img[data-src^='http'], img[src^='http']")
            for img_el in img_elements:
                try:
                    src = (
                        img_el.get_attribute("data-src")
                        or img_el.get_attribute("src")
                        or ""
                    )
                    if not src.startswith("http"):
                        continue
                    # Skip Google's own assets and tiny images.
                    if "gstatic.com" in src or "google.com" in src:
                        continue
                    if "encrypted-tbn" in src:
                        continue
                    if src in seen_urls:
                        continue
                    seen_urls.add(src)
                    results.append({
                        "title": img_el.get_attribute("alt") or query,
                        "page_url": "",
                        "image_url": src,
                        "original_url": src,
                        "width": None,
                        "height": None,
                        "license_name": None,
                        "license_url": None,
                        "attribution": img_el.get_attribute("alt") or query,
                    })
                except Exception:
                    continue

            # Extract from page source with regex patterns.
            for pattern in _IMG_URL_PATTERNS:
                for m in pattern.finditer(page_content):
                    img_url = m.group(1)
                    try:
                        w = int(m.group(2))
                        h = int(m.group(3))
                    except (IndexError, ValueError):
                        w, h = 0, 0

                    if not img_url.startswith("http"):
                        continue
                    # Skip Google's internal images.
                    if "gstatic.com" in img_url or "google.com" in img_url:
                        continue
                    if "encrypted-tbn" in img_url:
                        continue
                    if img_url in seen_urls:
                        continue
                    if min_width and w and w < min_width:
                        continue

                    seen_urls.add(img_url)
                    results.append({
                        "title": query,
                        "page_url": "",
                        "image_url": img_url,
                        "original_url": img_url,
                        "width": w or None,
                        "height": h or None,
                        "license_name": None,
                        "license_url": None,
                        "attribution": query,
                    })
                    if len(results) >= max_results:
                        break
                if len(results) >= max_results:
                    break

            # ── Strategy 2: Click thumbnails as last resort ──
            if len(results) < 3:
                thumbnails = page.query_selector_all(
                    "div[data-id] a[jsname], div.isv-r a.islib, "
                    "div.isv-r a[jsname], a[data-nav='1']"
                )
                max_clicks = min(len(thumbnails), max_results + 5)

                for idx in range(max_clicks):
                    if len(results) >= max_results:
                        break
                    try:
                        thumb = thumbnails[idx]
                        thumb.click(timeout=3000)
                        page.wait_for_timeout(1200)

                        img_el = (
                            page.query_selector("img.sFlh5c.pT0Scc.iPVvYb")
                            or page.query_selector("img[jsname='kn3ccd']")
                            or page.query_selector("img.r48jcc.pT0Scc.iPVvYb")
                            or page.query_selector("c-wiz img[src^='http']")
                        )
                        if not img_el:
                            continue
                        src = img_el.get_attribute("src") or ""
                        if not src.startswith("http"):
                            page.wait_for_timeout(1500)
                            src = img_el.get_attribute("src") or ""
                        if not src.startswith("http") or src in seen_urls:
                            continue
                        if "gstatic.com" in src or "google.com" in src:
                            continue
                        seen_urls.add(src)
                        results.append({
                            "title": img_el.get_attribute("alt") or query,
                            "page_url": "",
                            "image_url": src,
                            "original_url": src,
                            "width": None,
                            "height": None,
                            "license_name": None,
                            "license_url": None,
                            "attribution": img_el.get_attribute("alt") or query,
                        })
                    except Exception:
                        continue

            browser.close()

            if results:
                print(f"[playwright_images] Found {len(results)} images for {query!r}")
                return results

    except ImportError:
        print("[playwright_images] Playwright not installed — falling back to DuckDuckGo")
    except Exception as e:
        print(f"[playwright_images] Google Images scrape failed for {query!r}: {e}")

    # ── Fallback: DuckDuckGo ──
    return _ddg_image_fallback(query, max_results=max_results, min_width=min_width)


def _ddg_image_fallback(
    query: str,
    *,
    max_results: int = 20,
    min_width: int = 800,
) -> list[dict[str, Any]]:
    """Free image search via DuckDuckGo — no API key required."""
    try:
        from ddgs import DDGS

        d = DDGS()
        raw = d.images(query, max_results=max_results)

        out: list[dict[str, Any]] = []
        seen: set[str] = set()

        for item in raw:
            img_url = item.get("image", "")
            if not img_url or not img_url.startswith("http"):
                continue
            w = item.get("width", 0)
            if w and int(w) < min_width:
                continue
            if img_url in seen:
                continue
            seen.add(img_url)

            out.append({
                "title": item.get("title", ""),
                "page_url": item.get("url", ""),
                "image_url": img_url,
                "original_url": img_url,
                "width": item.get("width"),
                "height": item.get("height"),
                "license_name": None,
                "license_url": None,
                "attribution": item.get("source", item.get("title", "")),
            })
            if len(out) >= max_results:
                break

        if out:
            print(f"[playwright_images] DuckDuckGo found {len(out)} images for {query!r}")
        return out
    except Exception as e:
        print(f"[playwright_images] DuckDuckGo image search failed for {query!r}: {e}")
        return []
