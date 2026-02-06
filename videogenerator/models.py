from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Slide:
    start: float
    end: float
    image_path: str
    query: str
    source_page: str | None = None
    image_url: str | None = None
    license_name: str | None = None
    license_url: str | None = None
    attribution: str | None = None
