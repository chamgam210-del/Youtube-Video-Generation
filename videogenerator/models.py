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
    # Optional on-screen text (burned into slide cards for explainer/shorts formats).
    headline: str | None = None
    subhead: str | None = None
    source_page: str | None = None
    image_url: str | None = None
    license_name: str | None = None
    license_url: str | None = None
    attribution: str | None = None
    # Optional per-slide motion hint used by the renderer (e.g. hold/zoom_in/zoom_out/pan_lr/pan_rl/snap).
    motion: str | None = None

    # Debug-only fields written to timeline.json to help inspect image selection.
    window_text: str | None = None
    window_keywords: list[str] | None = None
    queries_tried: list[str] | None = None

    # When this slide should be replaced by a video clip during rendering.
    video_clip_path: str | None = None
    video_clip_start: float | None = None  # start offset inside the clip file
    video_clip_end: float | None = None  # end offset inside the clip file
    video_clip_mute: bool = False  # True → mute clip audio (copyright safety)


@dataclass(frozen=True)
class VideoClipSuggestion:
    """LLM-suggested video clip insertion point."""

    timeline_start: float  # seconds into the output video
    timeline_end: float
    search_query: str  # what to search for
    reason: str  # why this clip was suggested
    mute: bool = True  # default mute for copyright safety


@dataclass(frozen=True)
class CommentaryClipSuggestion:
    """LLM-suggested clip insertion for commentary videos.

    Two modes:
    - ``reference``: narrator references a specific clip ("here's the clip…").
      The system finds and inserts that one clip.
    - ``compilation``: narrator describes a broad reaction / news event.
      The system grabs several popular clips and concatenates them into a montage.
    """

    timeline_start: float
    timeline_end: float
    search_query: str  # primary search query
    reason: str
    clip_type: str = "reference"  # "reference" | "compilation"
    num_clips: int = 1  # how many clips to grab (>1 for compilation)
    extra_queries: list[str] | None = None  # additional search queries for compilations
    mute: bool = True
