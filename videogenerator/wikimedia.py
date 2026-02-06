from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .utils import sanitize_filename


_WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"

# Wikimedia asks API clients to send a descriptive User-Agent.
_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": "videogenerator-screenshots/0.1 (python; local)"
    }
)


def _ext(meta: dict[str, Any] | None, key: str) -> str | None:
    if not meta:
        return None
    v = meta.get(key)
    if isinstance(v, dict):
        return v.get("value")
    return None


def _license_ok(extmetadata: dict[str, Any] | None) -> bool:
    lic = (_ext(extmetadata, "LicenseShortName") or "").lower()
    usage = (_ext(extmetadata, "UsageTerms") or "").lower()
    if "all rights reserved" in lic or "all rights reserved" in usage:
        return False

    allowed = [
        "cc by",
        "cc-by",
        "cc by-sa",
        "cc-by-sa",
        "cc0",
        "public domain",
    ]
    return any(a in lic for a in allowed) or any(a in usage for a in allowed)


def _raster_ok(title: str | None, url: str | None) -> bool:
    s = (title or "") + " " + (url or "")
    s = s.lower()
    # Prefer common raster formats.
    return any(ext in s for ext in (".jpg", ".jpeg", ".png", ".webp"))


def _raster_url(url: str | None) -> bool:
    if not url:
        return False
    u = url.lower()
    return any(u.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp"))


def search_commons_images(
    query: str,
    *,
    min_width: int = 800,
    max_results: int = 12,
    timeout_s: int = 30,
) -> list[dict[str, Any]]:
    """Search Wikimedia Commons and return a ranked list of candidate images with license metadata."""

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": str(max_results),
        "gsrnamespace": "6",  # File:
        "prop": "imageinfo|info",
        "inprop": "url",
        "iiprop": "url|size|extmetadata",
        "iiurlwidth": "1600",
    }

    resp = _SESSION.get(_WIKIMEDIA_API, params=params, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

    pages = (data.get("query", {}) or {}).get("pages", {}) or {}
    candidates: list[dict[str, Any]] = []

    for _, page in pages.items():
        imageinfo = (page.get("imageinfo") or [None])[0]
        if not imageinfo:
            continue

        thumb_url = imageinfo.get("thumburl")
        orig_url = imageinfo.get("url")
        if not thumb_url and not orig_url:
            continue

        # Require the original URL to be a raster image; otherwise Commons often returns PDFs/DJVUs
        # whose thumbnails can dominate by size but aren't good slideshow images.
        if not _raster_url(orig_url):
            continue

        url = thumb_url or orig_url
        width = int(imageinfo.get("thumbwidth") or imageinfo.get("width") or 0)
        if width < min_width:
            continue

        if not _raster_ok(page.get("title"), url):
            continue

        extmetadata = imageinfo.get("extmetadata")
        if not _license_ok(extmetadata):
            continue

        candidates.append(
            {
                "title": page.get("title"),
                "page_url": page.get("fullurl") or page.get("canonicalurl"),
                "image_url": url,
                "original_url": orig_url,
                "width": width,
                "height": int(imageinfo.get("thumbheight") or imageinfo.get("height") or 0),
                "license_name": _ext(extmetadata, "LicenseShortName"),
                "license_url": _ext(extmetadata, "LicenseUrl"),
                "attribution": _ext(extmetadata, "Attribution")
                or _ext(extmetadata, "Artist")
                or _ext(extmetadata, "Credit"),
            }
        )

    candidates.sort(key=lambda c: (c.get("width", 0) * c.get("height", 0)), reverse=True)
    return candidates


def search_commons_image(
    query: str,
    *,
    min_width: int = 800,
    max_results: int = 12,
    timeout_s: int = 30,
) -> dict[str, Any] | None:
    """Search Wikimedia Commons and return a single best image dict with url + license metadata."""

    candidates = search_commons_images(
        query,
        min_width=min_width,
        max_results=max_results,
        timeout_s=timeout_s,
    )
    return candidates[0] if candidates else None


def download_image(image_url: str, dest_dir: str | Path, *, prefix: str = "img") -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    tmp_name = sanitize_filename(f"{prefix}_{int(time.time() * 1000)}") + ".bin"
    tmp_path = dest_dir / tmp_name

    # Use Commons API to resolve the provided URL to a stable original file URL.
    # This avoids thumbnail hotlink throttling (429).
    api_url = _WIKIMEDIA_API
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url",
    }

    # If we can extract a File: title from commons URLs, do so; otherwise use iiurl (still works for many urls).
    title = None
    if "commons.wikimedia.org/wiki/File:" in image_url:
        title = "File:" + image_url.split("commons.wikimedia.org/wiki/File:", 1)[1]
    elif "upload.wikimedia.org" in image_url:
        # Best-effort filename extraction
        from urllib.parse import urlparse, unquote

        name = Path(urlparse(image_url).path).name
        if name:
            title = "File:" + unquote(name.split("/", 1)[0])

    if title:
        params["titles"] = title
        try:
            meta = _SESSION.get(api_url, params=params, timeout=30).json()
            pages = (meta.get("query", {}) or {}).get("pages", {}) or {}
            for _, page in pages.items():
                ii = (page.get("imageinfo") or [None])[0]
                if ii and ii.get("url"):
                    image_url = ii["url"]
                    break
        except Exception:
            pass

    with _SESSION.get(image_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if ctype and not ctype.startswith("image/"):
            raise RuntimeError(f"Unexpected content-type for image: {ctype}")
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

    # Normalize to PNG for ffmpeg concat reliability.
    out_name = sanitize_filename(f"{prefix}_{int(time.time() * 1000)}") + ".png"
    out_path = dest_dir / out_name
    try:
        with Image.open(tmp_path) as im:
            im = im.convert("RGB")
            # Avoid PNG optimize=True: it can be extremely slow on some inputs
            # (and has triggered long-running encodes on Windows).
            im.save(out_path, format="PNG", compress_level=6)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return out_path
