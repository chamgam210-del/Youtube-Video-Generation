from __future__ import annotations

import json
from pathlib import Path

from .audio import get_audio_duration_seconds
from .audio_edit import concat_audio_clips_to_wav, write_highlight_clips_json
from .llm_storyboard import extract_review_highlights_with_llm
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
    max_slides: int = 4,
    target_total_seconds: float = 55.0,
) -> Path | None:
    """For a full-length review output, auto-generate a Shorts highlight video.

    Creates a `shorts_highlights/` subfolder inside review_out_dir.
    """

    review_audio_path = Path(review_audio_path)
    review_out_dir = Path(review_out_dir)

    transcript_json = review_out_dir / "transcript.json"
    if not transcript_json.exists():
        return None

    segments = _load_transcript_segments(transcript_json)
    if not segments:
        return None

    audio_duration = get_audio_duration_seconds(review_audio_path)

    # 1) Pick highlight windows (LLM)
    clips = extract_review_highlights_with_llm(
        segments,
        audio_duration=float(audio_duration),
        max_clips=int(max_slides),
        target_total_seconds=float(target_total_seconds),
        model=str(llm_model or "gpt-4o-mini"),
    )

    if not clips:
        return None

    shorts_dir = review_out_dir / "shorts_highlights"
    shorts_dir.mkdir(parents=True, exist_ok=True)

    write_highlight_clips_json(shorts_dir / "highlights.json", clips)

    # 2) Cut + concat audio into a Shorts-length WAV
    shorts_audio = concat_audio_clips_to_wav(
        review_audio_path,
        clips,
        out_wav=shorts_dir / "highlights.wav",
    )

    # 3) Run the existing Shorts pipeline on the shortened audio
    run_pipeline(
        audio_path=shorts_audio,
        out_dir=shorts_dir,
        topic=topic,
        video_type="shorts",
        image_provider=image_provider,
        serpapi_api_key=serpapi_api_key,
        max_images=int(max_slides),
        min_seg_seconds=6.0,
        whisper_model=whisper_model,
        min_image_width=int(min_image_width),
        video_width=1080,
        video_height=1920,
        cache_transcript=True,
        cache_dir=None,
        storyboard="llm",
        llm_model=str(llm_model or "gpt-4o-mini"),
        llm_pick_images=bool(llm_pick_images),
        reuse_images=bool(reuse_images),
    )

    # 4) Bake Shorts overlay + thumbnail, then render MP4
    slides = _load_slides(shorts_dir / "timeline.json")
    if not slides:
        return shorts_dir

    try:
        from .youtube import _try_find_raw_for_card, create_thumbnail, overlay_shorts_title_and_stamp

        # Thumbnail (stamp-only, raw background if possible)
        bg = Path(slides[0].image_path)
        raw = _try_find_raw_for_card(bg)
        if raw is not None:
            bg = raw

        create_thumbnail(
            out_path=shorts_dir / "thumbnail.png",
            background_image=bg,
            text="",
            verdict_text=None,
            stamp_text=None,
            match_video_frame=False,
            width=1080,
            height=1920,
            theme="highlight",
            show_title=False,
        )

        slides_to_render = overlay_shorts_title_and_stamp(
            slides,
            out_dir=shorts_dir / "slides_overlay",
            title="",
            stamp_text=None,
            width=1080,
            height=1920,
            show_title=False,
        )
    except Exception:
        slides_to_render = slides

    render_slideshow(
        slides_to_render,
        shorts_audio,
        shorts_dir / "video.mp4",
        width=1080,
        height=1920,
        fps=30,
        intro_seconds=0.0,
        outro_seconds=0.0,
        transition=None,
        transition_seconds=0.0,
        ken_burns=False,
        bgm_path=None,
        bgm_generate=False,
        bgm_preset=None,
    )

    return shorts_dir
