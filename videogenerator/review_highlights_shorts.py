from __future__ import annotations

import json
from pathlib import Path

from .audio import get_audio_duration_seconds
from .audio_edit import concat_audio_clips_to_wav, write_highlight_clips_json
from .llm_storyboard import extract_review_highlights_meaningful_with_llm
from .models import Slide, TranscriptSegment
from .pipeline import run as run_pipeline
from .render import render_slideshow


def _load_transcript_segments(transcript_json: Path) -> list[TranscriptSegment]:
    raw = json.loads(transcript_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    out: list[TranscriptSegment] = []
    for s in raw:
        try:
            out.append(TranscriptSegment(start=float(s["start"]), end=float(s["end"]), text=str(s["text"])))
        except Exception:
            continue
    return out


def _load_slides(timeline_json: Path) -> list[Slide]:
    raw = json.loads(timeline_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    out: list[Slide] = []
    for s in raw:
        try:
            out.append(Slide(**s))
        except Exception:
            continue
    return out


def make_shorts_from_review_highlights(
    *,
    review_audio_path: str | Path,
    review_out_dir: str | Path,
    topic: str | None,
    image_provider: str,
    serpapi_api_key: str | None,
    whisper_model: str,
    min_image_width: int,
    llm_model: str,
    llm_pick_images: bool,
    reuse_images: bool,
    bgm_path: str | Path | None = None,
    bgm_volume: float = 0.10,
    bgm_duck: bool = True,
    bgm_generate: bool = False,
    bgm_preset: str | None = None,
    max_slides: int = 4,
    target_total_seconds: float = 60.0,
) -> Path | None:
    """For a full-length review output, auto-generate a Shorts highlight video.

    Creates a `shorts_highlights/` subfolder inside review_out_dir.
    """

    # This feature is intentionally disabled for this project/workflow.
    # Long reviews should not auto-build a highlight short.
    return None
