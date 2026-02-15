from __future__ import annotations

import argparse
from pathlib import Path
import os
import json

from dotenv import load_dotenv

from .pipeline import run
from .render import ensure_bgm_preset_wav, render_slideshow
from .models import Slide


def main() -> None:
    # Load local secrets/config (e.g. SERPAPI_API_KEY) from `.env` if present.
    load_dotenv(override=False)

    p = argparse.ArgumentParser(description="MP3 -> transcript -> Wikimedia images -> MP4 slideshow")
    p.add_argument("--audio", required=False, help="Path to input audio (mp3/wav/etc)")
    p.add_argument("--out", default="output", help="Output folder")
    p.add_argument("--topic", default=None, help="Optional topic hint for image search, e.g. 'Severance TV series'")
    p.add_argument(
        "--visual-subject",
        default=None,
        help="Force image searches to target this exact movie/TV show name (useful when Shorts scripts don't say the title). Overrides --topic for image search.",
    )
    p.add_argument(
        "--image-provider",
        default="wikimedia",
        choices=["wikimedia", "serpapi", "google_images"],
        help=(
            "Where to search for images. "
            "'serpapi' uses Google Images via SerpAPI but is restricted to Wikimedia Commons hosts (license-aware). "
            "'google_images' uses SerpAPI Google Images results directly (no license validation)."
        ),
    )
    p.add_argument(
        "--serpapi-key",
        default=None,
        help="SerpAPI key (or set SERPAPI_API_KEY). Used when --image-provider serpapi/google_images.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=12,
        help="Max number of images to use (ignored for shorts_review; beat count is auto-determined)",
    )
    p.add_argument("--min-seg-seconds", type=float, default=6.0, help="Minimum transcript segment duration per slide")
    p.add_argument("--whisper-model", default="small", help="Whisper model name: tiny/base/small/medium/large")
    p.add_argument(
        "--storyboard",
        default="llm",
        choices=["auto", "llm", "heuristic"],
        help="How to decide slide cut points + queries. 'llm' requires OPENAI_API_KEY. 'auto' uses LLM if available, else falls back.",
    )
    p.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model for --storyboard llm/auto")

    p.add_argument(
        "--shorts-from-text",
        default=None,
        help="Path to a text file containing raw review thoughts. Generates 30-40s Shorts scripts and writes shorts_scripts.json into --out (no audio needed).",
    )
    p.add_argument(
        "--shorts-count",
        type=int,
        default=5,
        help="How many Shorts scripts to generate for --shorts-from-text (default: 5).",
    )
    p.add_argument(
        "--shorts-script-model",
        default="gpt-5.2",
        help="LLM model to use for --shorts-from-text (default: gpt-5.2).",
    )
    p.add_argument(
        "--no-llm-pick-images",
        action="store_true",
        help="When using LLM storyboard, disable LLM-based selection among image candidates (deterministic pick instead).",
    )
    p.add_argument("--no-transcript-cache", action="store_true", help="Disable transcript caching")
    p.add_argument(
        "--transcript-cache-dir",
        default=None,
        help="Override transcript cache directory (default: <out>/.cache/transcripts)",
    )
    p.add_argument("--min-image-width", type=int, default=900, help="Minimum image width (pixels)")
    p.add_argument("--width", type=int, default=1920, help="Video width")
    p.add_argument("--height", type=int, default=1080, help="Video height")
    p.add_argument("--fps", type=int, default=30, help="Video fps")

    p.add_argument(
        "--video-type",
        default="review",
        choices=["review", "explainer", "shorts", "shorts_review", "commentary", "auto"],
        help="Video style. 'review' uses image-only slides; 'explainer' uses text-on-slide cards; 'shorts_review' is retention-style short review (keyword cards, fast cuts); 'commentary' is for reaction/commentary videos with auto clip insertion. 'auto' tries to infer.",
    )
    p.add_argument(
        "--shorts",
        action="store_true",
        help="Preset for YouTube Shorts (9:16) with LLM-planned text slides.",
    )

    p.add_argument(
        "--transition",
        default="fade",
        choices=["none", "fade"],
        help="Simple slideshow transitions between images. 'fade' fades in/out each slide; 'none' disables transitions.",
    )
    p.add_argument(
        "--transition-seconds",
        type=float,
        default=0.35,
        help="Transition duration in seconds (used for fade).",
    )
    p.add_argument(
        "--ken-burns",
        action="store_true",
        help="Apply a subtle zoom/crop motion effect (Ken Burns).",
    )

    p.add_argument(
        "--bgm",
        default=None,
        help="Optional background music file (mp3/wav). Mixed quietly behind the narration.",
    )
    p.add_argument(
        "--bgm-volume",
        type=float,
        default=0.10,
        help="Background music volume multiplier (0.0-1.0). Default: 0.10.",
    )
    p.add_argument(
        "--bgm-generate",
        action="store_true",
        help="Generate a simple ambient background bed (no external audio file).",
    )
    p.add_argument(
        "--bgm-preset",
        type=str,
        default=None,
        help="Built-in background music preset: 'ambient', 'elevator', 'creepy', 'hiphop', 'rnb', or 'clown'.",
    )

    p.add_argument(
        "--write-bgm-wav",
        default=None,
        help="Generate a built-in BGM preset WAV into .cache/bgm_presets and print its path (no video render).",
    )
    p.add_argument(
        "--bgm-seconds",
        type=float,
        default=32.0,
        help="Seconds to generate for --write-bgm-wav (default: 32).",
    )
    p.add_argument(
        "--no-bgm-duck",
        action="store_true",
        help="Disable narration-aware ducking (sidechain compression) on background music.",
    )
    p.add_argument(
        "--mix-video-clips",
        action="store_true",
        help="Use AI to find and splice relevant video clips (B-roll) into the video. Requires SERPAPI_API_KEY and yt-dlp.",
    )
    p.add_argument(
        "--max-video-clips",
        type=int,
        default=6,
        help="Maximum number of video clips the AI can insert (default: 6).",
    )
    p.add_argument(
        "--clip-queries",
        nargs="+",
        default=None,
        help=(
            "Manual clip search queries for commentary videos. "
            "Each query becomes a clip that gets searched on YouTube. "
            "Example: --clip-queries 'Megyn Kelly reaction Bad Bunny' 'Ben Shapiro reaction Bad Bunny'"
        ),
    )
    p.add_argument(
        "--clip-research",
        action="store_true",
        default=False,
        help=(
            "Use agentic web research to find clips instead of simple YouTube search. "
            "Searches Google, scrapes news articles, and uses LLM to evaluate clips. "
            "Finds more accurate clips but uses more API calls. Requires playwright."
        ),
    )
    p.add_argument(
        "--verify-video",
        action="store_true",
        help="After render, verify audio presence and whether frames change",
    )

    p.add_argument(
        "--no-branding",
        action="store_true",
        help="Disable intro/outro branding slates (channel name, title, logo).",
    )
    p.add_argument(
        "--channel-name",
        default="Brutally Honest Review",
        help="Channel name to show on intro/outro slates.",
    )
    p.add_argument(
        "--logo-scheme",
        default="orange",
        choices=[
            "orange",
            "teal",
            "purple",
            "red",
            "slate",
            "lime",
            "mono",
            "orange_black",
            "black_orange",
            "black_orange_outline",
            "black_white_outline",
            "black_orange_flat",
        ],
        help="Brand logo color scheme. All variants are saved; this selects which one is used in intro/outro.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Override video title text (otherwise generated from transcript using the LLM when available).",
    )
    p.add_argument(
        "--intro-seconds",
        type=float,
        default=2.5,
        help="Seconds for the intro slate (narration is delayed by this amount).",
    )
    p.add_argument(
        "--outro-seconds",
        type=float,
        default=3.0,
        help="Seconds for the outro slate (pads with silence at end).",
    )
    p.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Only transcribe the audio and write transcript.json/transcript.txt into --out (no images, no video).",
    )

    p.add_argument(
        "--no-youtube-metadata",
        action="store_true",
        help="Disable generating YouTube title/description/tags + thumbnail assets in the output folder.",
    )
    p.add_argument(
        "--no-reuse-images",
        action="store_true",
        help="Disable reusing images from an existing output folder with the same audio stem.",
    )

    args = p.parse_args()

    if args.write_bgm_wav:
        wav = ensure_bgm_preset_wav(preset=str(args.write_bgm_wav), seconds=float(args.bgm_seconds))
        print(str(wav.resolve()))
        return

    if args.shorts_from_text:
        from .shorts_scripts import format_script_bracketed, generate_shorts_scripts_from_review_text_with_llm

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        src = Path(str(args.shorts_from_text))
        txt = src.read_text(encoding="utf-8")
        scripts = generate_shorts_scripts_from_review_text_with_llm(
            txt,
            topic=str((args.visual_subject or args.topic) or "").strip() or None,
            count=int(args.shorts_count),
            model=str(args.shorts_script_model),
        )

        payload = {
            "topic": str((args.visual_subject or args.topic) or "").strip(),
            "count": int(args.shorts_count),
            "model": str(args.shorts_script_model),
            "shorts": [
                {
                    "title": s.title,
                    "thumbnail_text": s.thumbnail_text,
                    "lines": [
                        {
                            "start": ln.start,
                            "end": ln.end,
                            "text": ln.text,
                            "keywords": ln.keywords,
                            "image_query": ln.image_query,
                        }
                        for ln in s.lines
                    ],
                }
                for s in scripts
            ],
        }

        (out_dir / "shorts_scripts.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        for i, s in enumerate(scripts, start=1):
            (out_dir / f"shorts_script_{i:02d}.txt").write_text(format_script_bracketed(s), encoding="utf-8")

        print(str((out_dir / "shorts_scripts.json").resolve()))
        return

    if not args.audio:
        raise SystemExit("--audio is required unless using --write-bgm-wav or --shorts-from-text")

    # Shorts preset overrides.
    vt_arg = str(getattr(args, "video_type", "")).strip().lower()
    is_shorts_like = args.shorts or (vt_arg in {"shorts", "shorts_review"})
    if is_shorts_like:
        # If user used --shorts, keep legacy behavior as "shorts".
        # If they explicitly chose shorts_review, preserve it.
        if args.shorts and vt_arg != "shorts_review":
            args.video_type = "shorts"
        else:
            args.video_type = (vt_arg or "shorts").strip().lower() or "shorts"
        # If user didn't explicitly override width/height, switch to 9:16.
        if int(args.width) == 1920 and int(args.height) == 1080:
            args.width = 1080
            args.height = 1920

        # Shorts retention defaults (only when user didn't explicitly override).
        import sys

        if "--max-images" not in sys.argv:
            args.max_images = 20
        if "--min-seg-seconds" not in sys.argv:
            args.min_seg_seconds = 1.8

        # Default: no transitions for shorts unless explicitly requested.
        if "--transition" not in sys.argv:
            args.transition = "none"
        if "--transition-seconds" not in sys.argv:
            args.transition_seconds = 0.0

        # Shorts should start immediately on content: disable intro/outro branding slates.
        args.no_branding = True
        args.intro_seconds = 0.0
        args.outro_seconds = 0.0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.transcribe_only:
        from .transcribe import transcribe_cached, write_transcript_files

        effective_cache_dir = (
            Path(args.transcript_cache_dir)
            if args.transcript_cache_dir
            else (Path.cwd() / ".cache" / "transcripts")
        )

        segments = transcribe_cached(
            args.audio,
            model_name=args.whisper_model,
            cache_dir=effective_cache_dir,
            use_cache=(not args.no_transcript_cache),
        )
        write_transcript_files(segments, out_dir=out_dir)
        print(str((out_dir / "transcript.txt").resolve()))
        return

    run(
        audio_path=args.audio,
        out_dir=out_dir,
        topic=(str(args.visual_subject).strip() or args.topic),
        video_type=str(args.video_type),
        image_provider=args.image_provider,
        serpapi_api_key=args.serpapi_key or os.getenv("SERPAPI_API_KEY"),
        max_images=args.max_images,
        min_seg_seconds=args.min_seg_seconds,
        whisper_model=args.whisper_model,
        min_image_width=args.min_image_width,
        video_width=int(args.width),
        video_height=int(args.height),
        cache_transcript=(not args.no_transcript_cache),
        cache_dir=args.transcript_cache_dir,
        storyboard=args.storyboard,
        llm_model=args.llm_model,
        llm_pick_images=(not args.no_llm_pick_images),
        reuse_images=(not args.no_reuse_images),
        mix_video_clips=bool(args.mix_video_clips),
        max_video_clips=int(args.max_video_clips),
        clip_queries=args.clip_queries,
        clip_research=bool(getattr(args, 'clip_research', False)),
    )

    # Load slides back from timeline.json for rendering
    timeline_path = out_dir / "timeline.json"
    data = json.loads(timeline_path.read_text(encoding="utf-8"))
    slides = [Slide(**s) for s in data]

    # Pipeline may produce a padded narration track (e.g., Shorts Review pivot silences).
    audio_for_render = args.audio
    try:
        meta_path = out_dir / "run_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rp = (meta or {}).get("render_audio_path")
            if rp:
                p = Path(str(rp))
                if p.exists() and p.stat().st_size > 4096:
                    audio_for_render = str(p)
    except Exception:
        audio_for_render = args.audio

    # Load transcript segments written by the pipeline (used for title generation).
    segments = None
    transcript_json = out_dir / "transcript.json"
    if transcript_json.exists():
        try:
            from .models import TranscriptSegment

            raw = json.loads(transcript_json.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                segments = [TranscriptSegment(start=float(s["start"]), end=float(s["end"]), text=str(s["text"])) for s in raw]
        except Exception:
            segments = None

    intro_seconds = max(0.0, float(args.intro_seconds))
    outro_seconds = max(0.0, float(args.outro_seconds))

    # Generate a title based on the transcript (LLM when available).
    channel_name = str(args.channel_name or "").strip() or "Brutally Honest Review"
    title = str(args.title or "").strip()
    vt = (str(args.video_type) or "review").strip().lower()
    cleaned_topic_title = ""
    shorts_overlay_stamp: str | None = None
    if not title:
        # Prefer a non-spoilery, non-revealing generic title when topic is provided.
        if args.topic:
            topic = str(args.topic)
            # Light normalization for common patterns.
            topic = topic.replace("(TV series)", "").replace("TV series", "").strip(" -|:")
            cleaned_topic_title = topic
            # If the user provides a topic hint, use it as the title (clean + consistent).
            title = cleaned_topic_title

        from .llm_storyboard import generate_video_title_with_llm, generate_video_title_fallback

        if (not title) and segments:
            try:
                title = generate_video_title_with_llm(
                    segments,
                    topic=args.topic,
                    channel_name=channel_name,
                    video_type=vt,
                    model=args.llm_model,
                )
            except Exception:
                title = ""

        if not title:
            title = generate_video_title_fallback(args.audio, topic=args.topic, video_type=vt)

    try:
        (out_dir / "title.txt").write_text(title + "\n", encoding="utf-8")
    except Exception:
        pass

    if not args.no_youtube_metadata:
        try:
            from .youtube import (
                YouTubePackage,
                create_thumbnail,
                generate_youtube_package,
                pick_long_review_thumbnail_with_vision,
                pick_review_thumbnail_text_with_llm,
                write_youtube_metadata_text,
            )

            pkg = generate_youtube_package(
                segments,
                slides=slides,
                topic=args.topic,
                channel_name=channel_name,
                title=title,
                video_type=str(args.video_type),
                model=args.llm_model,
            )

            # Long review thumbnails: use vision to select a face-forward still + tight crop + 2–4 word hook.
            if vt == "review":
                try:
                    forced_phrase = None
                    try:
                        if segments:
                            forced_phrase = pick_review_thumbnail_text_with_llm(
                                segments,
                                title=title,
                                model=args.llm_model,
                            )
                    except Exception:
                        forced_phrase = None
                    idx_v, thumb_text, thumb_crop = pick_long_review_thumbnail_with_vision(
                        slides=slides,
                        topic=args.topic,
                        title=title,
                        model=args.llm_model,
                        forced_text=forced_phrase,
                    )
                    pkg = YouTubePackage(
                        title=pkg.title,
                        description=pkg.description,
                        tags=pkg.tags,
                        thumbnail_slide_index=int(idx_v),
                        verdict_label=pkg.verdict_label,
                        thumbnail_stamp_text=None,
                        thumbnail_text=thumb_text,
                        thumbnail_crop=thumb_crop,
                    )
                except Exception:
                    pass
            write_youtube_metadata_text(out_dir, pkg)
            shorts_overlay_stamp = pkg.thumbnail_stamp_text

            # Shorts Review: skip separate thumbnail asset generation (captions/keywords only workflow).
            if vt != "shorts_review":
                # Thumbnail background comes from a chosen slide asset.
                thumb_slide = slides[max(0, min(len(slides) - 1, int(pkg.thumbnail_slide_index)))]
                thumb_path = out_dir / "thumbnail.png"

                bg_img = thumb_slide.image_path
                # For explainer/shorts, slides may be rendered card_XX.png with LLM text.
                # For Shorts thumbnails, prefer the raw downloaded image (sXX_*.png).
                if vt in {"shorts"}:
                    try:
                        from .youtube import _try_find_raw_for_card
                        from pathlib import Path as _P

                        raw = _try_find_raw_for_card(_P(bg_img))
                        if raw is not None:
                            bg_img = str(raw)
                    except Exception:
                        pass
                create_thumbnail(
                    out_path=thumb_path,
                    background_image=bg_img,
                    text=((pkg.title or title) if vt == "review" else (cleaned_topic_title or ("Brutally Honest Review" if vt == "review" else str(pkg.title or title)))),
                    verdict_text=(pkg.verdict_label if vt == "review" else pkg.verdict_label),
                    stamp_text=(None if vt == "review" else pkg.thumbnail_stamp_text),
                    match_video_frame=False,
                    width=(int(args.width) if vt in {"shorts"} else 1280),
                    height=(int(args.height) if vt in {"shorts"} else 720),
                    theme=("highlight" if vt in {"shorts"} else ("review_long" if vt == "review" else "default")),
                    show_title=(True if vt == "review" else (vt not in {"shorts"})),
                    crop=(pkg.thumbnail_crop if vt == "review" else None),
                )
        except Exception:
            pass

    # Shorts: keep a persistent title+stamp overlay throughout the entire video.
    if vt in {"shorts"} and slides:
        try:
            from .youtube import overlay_shorts_title_and_stamp

            overlay_title = cleaned_topic_title or title
            slides = overlay_shorts_title_and_stamp(
                slides,
                out_dir=out_dir / "slides_overlay",
                title=overlay_title,
                stamp_text=shorts_overlay_stamp,
                width=int(args.width),
                height=int(args.height),
                show_title=False,
            )
        except Exception:
            pass

    # Shorts Review: audio starts after the hook frame (hook is produced by the pipeline timeline).
    hook_s = 4.0 if (vt == "shorts_review") else 0.0

    if not args.no_branding and (intro_seconds > 0.0 or outro_seconds > 0.0):

        from .branding import create_branding_assets

        assets = create_branding_assets(
            out_dir=out_dir,
            width=int(args.width),
            height=int(args.height),
            channel_name=channel_name,
            title=title,
            logo_scheme=args.logo_scheme,
        )

        branded: list[Slide] = []
        t = 0.0
        if intro_seconds > 0.0:
            branded.append(
                Slide(
                    start=0.0,
                    end=float(intro_seconds),
                    image_path=str(assets["intro"].as_posix()),
                    query="intro_slate",
                )
            )
            t = float(intro_seconds)

        # Shift existing slides forward to make room for intro slate.
        for s in slides:
            branded.append(
                Slide(
                    start=float(s.start) + t,
                    end=float(s.end) + t,
                    image_path=s.image_path,
                    query=s.query,
                    source_page=s.source_page,
                    image_url=s.image_url,
                    license_name=s.license_name,
                    license_url=s.license_url,
                    attribution=s.attribution,
                    motion=getattr(s, "motion", None),
                    video_clip_path=getattr(s, "video_clip_path", None),
                    video_clip_start=getattr(s, "video_clip_start", None),
                    video_clip_end=getattr(s, "video_clip_end", None),
                    video_clip_mute=getattr(s, "video_clip_mute", False),
                )
            )

        if outro_seconds > 0.0:
            end_t = float(branded[-1].end) if branded else t
            branded.append(
                Slide(
                    start=end_t,
                    end=end_t + float(outro_seconds),
                    image_path=str(assets["outro"].as_posix()),
                    query="outro_slate",
                )
            )

        slides = branded

    out_mp4 = out_dir / "video.mp4"

    bgm_path = args.bgm
    bgm_generate = bool(args.bgm_generate)
    bgm_preset = args.bgm_preset
    if bgm_path and (bgm_generate or bgm_preset):
        raise SystemExit("Provide only one of --bgm, --bgm-generate, or --bgm-preset")
    if bgm_preset and bgm_generate:
        # Treat --bgm-preset as the explicit choice; --bgm-generate becomes redundant.
        bgm_generate = False

    # Auto rule: for garbage reviews, default to clown BGM (unless user chose a BGM option).
    if (not bgm_path) and (not bgm_generate) and (not bgm_preset) and vt == "review" and segments:
        try:
            from .youtube import _fallback_verdict_label

            if _fallback_verdict_label(segments=segments) == "Garbage!":
                bgm_preset = "clown"
        except Exception:
            pass

    render_slideshow(
        slides,
        audio_for_render,
        out_mp4,
        width=args.width,
        height=args.height,
        fps=args.fps,
        bgm_path=bgm_path,
        bgm_volume=args.bgm_volume,
        bgm_duck=(not args.no_bgm_duck),
        bgm_generate=bgm_generate,
        bgm_preset=bgm_preset,
        intro_seconds=(float(hook_s) if vt == "shorts_review" else (intro_seconds if (not args.no_branding) else 0.0)),
        outro_seconds=outro_seconds if (not args.no_branding) else 0.0,
        transition=None if args.transition == "none" else args.transition,
        transition_seconds=float(args.transition_seconds),
        ken_burns=(vt == "shorts_review") or bool(args.ken_burns),
    )

    if args.verify_video:
        from .verify_video import verify_local

        v = verify_local(out_mp4)
        print(v.duration_line.strip())
        print(f"has_audio={v.has_audio} audio_peak={v.audio_peak} frame_hashes={v.frame_hashes}")

    print(str(out_mp4.resolve()))


if __name__ == "__main__":
    main()
