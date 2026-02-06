from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .serpapi_images import filter_allowlisted_hosts, serpapi_google_images
from .wikimedia import search_commons_image


_ALLOW_HOSTS = {
    "commons.wikimedia.org",
    "upload.wikimedia.org",
}


def _filename_from_upload_url(url: str) -> str | None:
    # Typical: https://upload.wikimedia.org/wikipedia/commons/.../Some_File_Name.jpg
    try:
        path = urlparse(url).path
        if not path:
            return None
        name = Path(path).name
        if not name or "." not in name:
            return None
        return unquote(name)
    except Exception:
        return None


def _file_title_from_commons_link(url: str | None) -> str | None:
    if not url:
        return None
    try:
        u = urlparse(url)
        if (u.hostname or "").lower() != "commons.wikimedia.org":
            return None
        path = u.path or ""
        if "/wiki/File:" in path:
            return "File:" + unquote(path.split("/wiki/File:", 1)[1])
        return None
    except Exception:
        return None


def _best_file_title(img_original: str | None, img_link: str | None) -> str | None:
    # Prefer an explicit Commons file page if available.
    t = _file_title_from_commons_link(img_link)
    if t:
        return t
    # Next, try to recover from upload.wikimedia.org URLs.
    for u in (img_original, img_link):
        if not u:
            continue
        fn = _filename_from_upload_url(u)
        if fn:
            return f"File:{fn}"
    return None


def search_commons_via_serpapi(
    query: str,
    *,
    api_key: str,
    min_width: int = 800,
    max_results: int = 10,
) -> dict[str, Any] | None:
    """Use SerpAPI Google Images, but restrict results to Wikimedia Commons.

    We still use the Commons API for license metadata + filtering.
    """

    # Constrain to Commons hosts to avoid copyright issues.
    q = f"{query} site:commons.wikimedia.org OR site:upload.wikimedia.org"

    imgs = serpapi_google_images(q, api_key=api_key, num=max_results)
    imgs = filter_allowlisted_hosts(imgs, _ALLOW_HOSTS)

    # Best effort: if we got an upload.wikimedia.org URL or a commons file page link, resolve to a commons File: title.
    for img in imgs:
        title = _best_file_title(img.original, img.link)
        if not title:
            continue
        info = search_commons_image(title, min_width=min_width, max_results=5)
        if info:
            return info

    # Fallback to normal commons search with the original query.
    return search_commons_image(query, min_width=min_width, max_results=max_results)


def search_commons_candidates_via_serpapi(
    query: str,
    *,
    api_key: str,
    min_width: int = 800,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Return multiple Commons candidates discovered via SerpAPI.

    Each candidate is a Commons info dict (license-validated) as returned by `search_commons_image`.
    """

    q = f"{query} site:commons.wikimedia.org OR site:upload.wikimedia.org"
    imgs = serpapi_google_images(q, api_key=api_key, num=max_results)
    imgs = filter_allowlisted_hosts(imgs, _ALLOW_HOSTS)

    seen_pages: set[str] = set()
    out: list[dict[str, Any]] = []

    for img in imgs:
        title = _best_file_title(img.original, img.link)
        if not title:
            continue
        info = search_commons_image(title, min_width=min_width, max_results=5)
        if not info:
            continue
        page = str(info.get("page_url") or "")
        if page and page in seen_pages:
            continue
        if page:
            seen_pages.add(page)
        out.append(info)
        if len(out) >= max_results:
            break

    # If SerpAPI didn't produce candidates, fall back to Commons search so the pipeline can proceed.
    if not out:
        info = search_commons_image(query, min_width=min_width, max_results=max_results)
        if info:
            out.append(info)

    return out


def search_google_images_candidates_via_serpapi(
    query: str,
    *,
    api_key: str,
    max_results: int = 12,
    min_width: int = 800,
    safe: str = "active",
) -> list[dict[str, Any]]:
    """Return candidates directly from SerpAPI Google Images.

    This mode is not restricted to Wikimedia Commons, so we apply a strict
    usage-rights filter at the API level.

    Notes:
    - We also drop shopping/product results.
    """

    imgs = serpapi_google_images(
        query,
        api_key=api_key,
        num=max_results,
        safe=safe,
        image_type="photo",
    )

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for img in imgs:
        if img.is_product:
            continue
        if not img.original or not (img.original.startswith("http://") or img.original.startswith("https://")):
            continue
        if img.original.startswith("x-raw-image:"):
            continue
        if img.original_width and int(img.original_width) < int(min_width):
            continue
        if img.original in seen:
            continue
        seen.add(img.original)

        out.append(
            {
                "title": img.title,
                "page_url": img.link,
                "image_url": img.original,
                "original_url": img.original,
                "width": img.original_width,
                "height": img.original_height,
                "license_name": None,
                "license_url": None,
                "attribution": img.source or img.title,
            }
        )
        if len(out) >= max_results:
            break

    return out
