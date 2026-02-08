from __future__ import annotations

import argparse
from pathlib import Path
import os
import json

from dotenv import load_dotenv

from .pipeline import run
from .render import render_slideshow
from .models import Slide


def main() -> None:
    # Load local secrets/config (e.g. SERPAPI_API_KEY) from `.env` if present.
    load_dotenv(override=False)

    p = argparse.ArgumentParser(description="MP3 -> transcript -> Wikimedia images -> MP4 slideshow")
    p.add_argument("--audio", required=True, help="Path to input audio (mp3/wav/etc)")
    p.add_argument("--out", default="output", help="Output folder")
    p.add_argument("--topic", default=None, help="Optional topic hint for image search, e.g. 'Severance TV series'")
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
    p.add_argument("--max-images", type=int, default=12, help="Max number of images to use")
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
        choices=["review", "explainer", "shorts", "auto"],
        help="Video style. 'review' uses image-only slides; 'explainer' uses text-on-slide cards; 'auto' tries to infer.",
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
        help="(Disabled) Previously applied a zoom effect; kept for compatibility but currently has no effect.",
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
        help="Built-in background music preset: 'ambient', 'elevator', or 'creepy'.",
    )
    p.add_argument(
        "--no-bgm-duck",
        action="store_true",
        help="Disable narration-aware ducking (sidechain compression) on background music.",
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

    # Shorts preset overrides.
    if args.shorts or str(getattr(args, "video_type", "")).strip().lower() == "shorts":
        args.video_type = "shorts"
        # If user didn't explicitly override width/height, switch to 9:16.
        if int(args.width) == 1920 and int(args.height) == 1080:
            args.width = 1080
            args.height = 1920

        # Shorts should be punchy: cap slides.
        try:
            args.max_images = min(int(args.max_images), 4)
        except Exception:
            args.max_images = 4

        # Default: no transitions for shorts unless explicitly requested.
        import sys

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
        topic=args.topic,
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
    )

    # Load slides back from timeline.json for rendering
    timeline_path = out_dir / "timeline.json"
    data = json.loads(timeline_path.read_text(encoding="utf-8"))
    slides = [Slide(**s) for s in data]

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
            from .youtube import create_thumbnail, generate_youtube_package, write_youtube_metadata_text

            pkg = generate_youtube_package(
                segments,
                slides=slides,
                topic=args.topic,
                channel_name=channel_name,
                title=title,
                video_type=str(args.video_type),
                model=args.llm_model,
            )
            write_youtube_metadata_text(out_dir, pkg)
            shorts_overlay_stamp = pkg.thumbnail_stamp_text

            # Thumbnail background comes from a chosen slide asset.
            thumb_slide = slides[max(0, min(len(slides) - 1, int(pkg.thumbnail_slide_index)))]
            thumb_path = out_dir / "thumbnail.png"

            bg_img = thumb_slide.image_path
            # For explainer/shorts, slides may be rendered card_XX.png with LLM text.
            # For Shorts thumbnails, prefer the raw downloaded image (sXX_*.png).
            if vt == "shorts":
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
                text=(cleaned_topic_title or ("Brutally Honest Review" if vt == "review" else str(pkg.title or title))),
                verdict_text=pkg.verdict_label,
                stamp_text=pkg.thumbnail_stamp_text,
                match_video_frame=False,
                width=(int(args.width) if vt == "shorts" else 1280),
                height=(int(args.height) if vt == "shorts" else 720),
                theme=("highlight" if vt == "shorts" else "default"),
                show_title=(vt != "shorts"),
            )
        except Exception:
            pass

    # Shorts: keep a persistent title+stamp overlay throughout the entire video.
    if vt == "shorts" and slides:
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

    render_slideshow(
        slides,
        args.audio,
        out_mp4,
        width=args.width,
        height=args.height,
        fps=args.fps,
        bgm_path=bgm_path,
        bgm_volume=args.bgm_volume,
        bgm_duck=(not args.no_bgm_duck),
        bgm_generate=bgm_generate,
        bgm_preset=bgm_preset,
        intro_seconds=intro_seconds if (not args.no_branding) else 0.0,
        outro_seconds=outro_seconds if (not args.no_branding) else 0.0,
        transition=None if args.transition == "none" else args.transition,
        transition_seconds=float(args.transition_seconds),
        ken_burns=bool(args.ken_burns),
    )

    if args.verify_video:
        from .verify_video import verify_local

        v = verify_local(out_mp4)
        print(v.duration_line.strip())
        print(f"has_audio={v.has_audio} audio_peak={v.audio_peak} frame_hashes={v.frame_hashes}")

    print(str(out_mp4.resolve()))


if __name__ == "__main__":
    main()
