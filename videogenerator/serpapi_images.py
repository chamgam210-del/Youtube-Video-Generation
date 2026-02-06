from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests


_SERPAPI_ENDPOINT = "https://serpapi.com/search.json"


@dataclass(frozen=True)
class SerpImage:
    original: str
    link: str | None = None
    source: str | None = None
    title: str | None = None
    is_product: bool | None = None
    original_width: int | None = None
    original_height: int | None = None


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def serpapi_google_images(
    query: str,
    *,
    api_key: str,
    num: int = 10,
    google_domain: str = "google.com",
    hl: str = "en",
    gl: str = "us",
    safe: str = "active",
    ijn: int = 0,
    image_type: str | None = None,
    timeout_s: int = 30,
) -> list[SerpImage]:
    params = {
        "engine": "google_images",
        "q": query,
        "google_domain": google_domain,
        "hl": hl,
        "gl": gl,
        "safe": safe,
        "ijn": str(ijn),
        "api_key": api_key,
    }

    # Advanced filters.
    # See: https://serpapi.com/google-images-api
    if image_type:
        params["image_type"] = image_type

    resp = requests.get(_SERPAPI_ENDPOINT, params=params, timeout=timeout_s)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()

    results = data.get("images_results") or []
    out: list[SerpImage] = []

    for r in results[: max(1, num)]:
        original = r.get("original") or r.get("thumbnail")
        if not original:
            continue
        out.append(
            SerpImage(
                original=str(original),
                link=r.get("link"),
                source=r.get("source"),
                title=r.get("title"),
                is_product=r.get("is_product"),
                original_width=(int(r.get("original_width")) if r.get("original_width") else None),
                original_height=(int(r.get("original_height")) if r.get("original_height") else None),
            )
        )

    return out


def filter_allowlisted_hosts(images: list[SerpImage], allow_hosts: set[str]) -> list[SerpImage]:
    out: list[SerpImage] = []
    for img in images:
        if _host(img.original) in allow_hosts or (img.link and _host(img.link) in allow_hosts):
            out.append(img)
    return out
