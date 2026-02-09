from __future__ import annotations

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

import streamlit as st

from dotenv import load_dotenv

from videogenerator.pipeline import run
from videogenerator.render import render_slideshow
from videogenerator.models import Slide
from videogenerator.verify_video import verify_local
from videogenerator.youtube import create_thumbnail, generate_youtube_package, write_youtube_metadata_text


def _default_output_dir(audio_path: Path) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = audio_path.stem if audio_path else "run"
    return f"output_{stem}_{ts}"


def _load_slides(out_dir: Path) -> list[Slide]:
    timeline_path = out_dir / "timeline.json"
    data = json.loads(timeline_path.read_text(encoding="utf-8"))
    return [Slide(**s) for s in data]


def _open_folder(path: Path) -> None:
    # Best-effort local convenience.
    try:
        path = Path(path).resolve()
        if sys.platform.startswith("win"):
            # explorer.exe tends to be the most reliable for folders.
            try:
                import subprocess

                subprocess.Popen(["explorer", str(path)])
            except Exception:
                os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess

            subprocess.Popen(["open", str(path)])
        else:
            import subprocess

            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def _clean_topic_for_title(topic: str) -> str:
    t = (topic or "").strip()
    t = t.replace("(TV series)", "").replace("TV series", "").strip(" -|:")
    return t


st.set_page_config(page_title="VideoGenerator", layout="wide")

# Load local secrets/config (e.g. SERPAPI_API_KEY, OPENAI_API_KEY) from `.env` if present.
load_dotenv(override=False)

st.title("MP3 → Transcript → Images → Video")
st.caption("Local UI for running the videogenerator pipeline")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1) Pick an MP3")
    audio_file = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a", "aac", "flac", "ogg"])

    st.subheader("2) Key settings")
    # Default topic hint from the uploaded filename to avoid stale defaults.
    if audio_file is not None:
        last_name = st.session_state.get("_last_audio_name")
        if last_name != audio_file.name:
            stem = Path(audio_file.name).stem
            suggested = stem.replace("_", " ").replace("-", " ").strip()
            st.session_state["topic_hint"] = suggested
            st.session_state["_last_audio_name"] = audio_file.name

    topic = st.text_input(
        "Topic hint (optional)",
        key="topic_hint",
        help="Used as an image-search hint and for metadata only. Leave blank if unsure.",
    )

    image_provider = st.selectbox(
        "Image provider",
        options=["google_images", "serpapi", "wikimedia"],
        index=0,
        help="google_images uses SerpAPI Google Images results directly (no license validation).",
    )

    video_type = st.selectbox(
        "Video type",
        options=["review (images only)", "explainer (text + images)", "shorts (9:16)", "auto"],
        index=0,
        help="Explainer/shorts use LLM-planned text-on-slide storyboards when available.",
    )

    max_images = st.slider("Max images", min_value=4, max_value=24, value=12, step=1)
    min_seg_seconds = st.slider("Min seconds per slide", min_value=3.0, max_value=12.0, value=6.0, step=0.5)

    transition = st.selectbox("Transition", options=["fade", "none"], index=0)
    transition_seconds = st.slider("Transition seconds", min_value=0.0, max_value=1.0, value=0.35, step=0.05)

    st.subheader("3) Audio mix")
    bgm_preset = st.selectbox("BGM preset", options=["elevator", "ambient", "creepy", "(none)"], index=0)
    bgm_volume = st.slider("BGM volume", min_value=0.0, max_value=0.30, value=0.16, step=0.01)
    bgm_duck = st.checkbox("Ducking (reduce BGM under narration)", value=True)

    st.subheader("4) Branding")
    channel_name = st.text_input("Channel name", value="Brutally Honest Review")
    title_override = st.text_input(
        "Video title (optional override)",
        value="",
        help="Leave blank to auto-generate from transcript/topic.",
    )
    reuse_existing_title_txt = st.checkbox(
        "Reuse existing title.txt if present",
        value=False,
        help="If unchecked, the UI recomputes the title each run (recommended if you changed topic).",
    )
    intro_seconds = st.slider("Intro seconds", min_value=0.0, max_value=6.0, value=2.5, step=0.5)
    outro_seconds = st.slider("Outro seconds", min_value=0.0, max_value=8.0, value=3.0, step=0.5)

    reuse_images = st.checkbox("Reuse images across reruns", value=True)
    youtube_metadata = st.checkbox("Generate YouTube metadata + thumbnail", value=True)
    verify_video = st.checkbox("Verify MP4 (local)", value=True)

with col_right:
    st.subheader("Output")

    workspace = Path.cwd()

    out_name = st.text_input(
        "Output folder name",
        value="",
        help="Leave blank to auto-name a fresh output folder.",
    )

    if audio_file is None:
        st.info("Upload an audio file to enable Run.")
        st.stop()

    tmp_dir = workspace / ".cache" / "ui_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    saved_audio = tmp_dir / audio_file.name

    # IMPORTANT: transcript caching keys on (audio_name, size, mtime).
    # Streamlit's uploader provides bytes each rerun; if we rewrite the file every time,
    # its mtime changes and we will re-transcribe. So: only write when content changed.
    uploaded_bytes = audio_file.getvalue()
    uploaded_sha = hashlib.sha256(uploaded_bytes).hexdigest()
    sha_path = tmp_dir / f"{audio_file.name}.sha256"

    prev_sha = ""
    if sha_path.exists():
        try:
            prev_sha = sha_path.read_text(encoding="utf-8").strip()
        except Exception:
            prev_sha = ""

    if (not saved_audio.exists()) or (prev_sha != uploaded_sha):
        saved_audio.write_bytes(uploaded_bytes)
        try:
            sha_path.write_text(uploaded_sha + "\n", encoding="utf-8")
        except Exception:
            pass

    if not out_name.strip():
        out_name = _default_output_dir(saved_audio)

    out_dir = workspace / out_name
    st.code(str(out_dir), language="text")

    # Show whether we have prior outputs for this MP3 stem (image reuse).
    stem = saved_audio.stem.strip().lower()
    if stem:
        prior = []
        for d in workspace.iterdir():
            if not d.is_dir():
                continue
            nm = d.name.lower()
            if not nm.startswith("output"):
                continue
            if d.resolve() == out_dir.resolve():
                continue
            if stem not in nm:
                continue
            if not (d / "timeline.json").exists():
                continue
            prior.append(d)
        if prior:
            prior.sort(key=lambda p: (p / "timeline.json").stat().st_mtime if (p / "timeline.json").exists() else 0.0, reverse=True)
            st.info(f"Found {len(prior)} prior output folder(s) for '{saved_audio.stem}'. Latest: {prior[0].name}")
        else:
            st.caption("No prior output folders detected for this MP3 stem (yet).")

    run_clicked = st.button("Run", type="primary")

    if run_clicked:
        out_dir.mkdir(parents=True, exist_ok=True)

        vt = "review"
        if video_type.startswith("explainer"):
            vt = "explainer"
        elif video_type.startswith("shorts"):
            vt = "shorts"
        elif video_type == "auto":
            vt = "auto"

        # Render sizing preset.
        vid_w, vid_h = (1920, 1080)
        if vt == "shorts":
            vid_w, vid_h = (1080, 1920)

        # Shorts defaults: max 4 slides, no transitions.
        run_max_images = int(max_images)
        run_transition = transition
        run_transition_seconds = float(transition_seconds)
        if vt == "shorts":
            run_max_images = min(run_max_images, 4)
            run_transition = "none"
            run_transition_seconds = 0.0

        # Pipeline
        with st.status("Running pipeline…", expanded=True) as status:
            st.write("Planning slides, searching images, writing timeline…")
            run(
                audio_path=str(saved_audio),
                out_dir=out_dir,
                topic=topic or None,
                video_type=vt,
                image_provider=image_provider,
                serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
                max_images=int(run_max_images),
                min_seg_seconds=float(min_seg_seconds),
                whisper_model="small",
                min_image_width=900,
                video_width=int(vid_w),
                video_height=int(vid_h),
                cache_transcript=True,
                cache_dir=None,
                storyboard="auto",
                llm_model="gpt-4o-mini",
                llm_pick_images=True,
                reuse_images=bool(reuse_images),
            )

            # Show reuse decision (if any) from pipeline metadata.
            try:
                meta_path = out_dir / "run_meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    reused_from = meta.get("reused_from")
                    if reused_from:
                        st.success(f"Reused images from: {reused_from}")
                    else:
                        st.caption("Did not reuse images (no matching prior output found).")
            except Exception:
                pass

            slides = _load_slides(out_dir)

            # Load transcript segments for title + YouTube packaging.
            segments = None
            transcript_json = out_dir / "transcript.json"
            if transcript_json.exists():
                try:
                    from videogenerator.models import TranscriptSegment

                    raw = json.loads(transcript_json.read_text(encoding="utf-8"))
                    if isinstance(raw, list):
                        segments = [
                            TranscriptSegment(start=float(s["start"]), end=float(s["end"]), text=str(s["text"]))
                            for s in raw
                        ]
                except Exception:
                    segments = None

            # Compute title text (override -> topic-based -> LLM -> fallback; optionally reuse existing title.txt).
            title_txt = (title_override or "").strip()
            if not title_txt:
                if reuse_existing_title_txt:
                    try:
                        title_txt = (out_dir / "title.txt").read_text(encoding="utf-8").strip()
                    except Exception:
                        title_txt = ""

            if not title_txt and (topic or "").strip():
                cleaned = _clean_topic_for_title(str(topic))
                if cleaned:
                    # If user provides a topic hint, use it as the title (clean + consistent).
                    title_txt = cleaned

            if not title_txt and segments:
                try:
                    from videogenerator.llm_storyboard import generate_video_title_with_llm

                    title_txt = generate_video_title_with_llm(
                        segments,
                        topic=topic or None,
                        channel_name=str(channel_name or "").strip() or "Brutally Honest Review",
                        video_type=vt,
                        model="gpt-4o-mini",
                    )
                except Exception:
                    title_txt = ""

            if not title_txt:
                try:
                    from videogenerator.llm_storyboard import generate_video_title_fallback

                    title_txt = generate_video_title_fallback(str(saved_audio), topic=topic or None, video_type=vt)
                except Exception:
                    if vt == "review":
                        title_txt = f"Brutally Honest Review of {saved_audio.stem}".strip()
                    else:
                        title_txt = saved_audio.stem.replace("_", " ").strip()

            try:
                (out_dir / "title.txt").write_text(title_txt + "\n", encoding="utf-8")
            except Exception:
                pass

            # Branding slates (intro/outro) based on channel name + title.
            slides_to_render = slides
            intro_s = max(0.0, float(intro_seconds))
            outro_s = max(0.0, float(outro_seconds))

            # Shorts: no channel intro/outro. Start immediately on slide 0.
            if vt == "shorts":
                intro_s = 0.0
                outro_s = 0.0

            if (vt != "shorts") and (intro_s > 0.0 or outro_s > 0.0):
                try:
                    from videogenerator.branding import create_branding_assets

                    assets = create_branding_assets(
                        out_dir=out_dir,
                        width=1920,
                        height=1080,
                        channel_name=str(channel_name or "").strip() or "Brutally Honest Review",
                        title=title_txt,
                        logo_scheme="orange",
                    )

                    branded: list[Slide] = []
                    t = 0.0
                    if intro_s > 0.0:
                        branded.append(
                            Slide(
                                start=0.0,
                                end=float(intro_s),
                                image_path=str(assets["intro"].as_posix()),
                                query="intro_slate",
                            )
                        )
                        t = float(intro_s)

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

                    if outro_s > 0.0:
                        end_t = float(branded[-1].end) if branded else t
                        branded.append(
                            Slide(
                                start=end_t,
                                end=end_t + float(outro_s),
                                image_path=str(assets["outro"].as_posix()),
                                query="outro_slate",
                            )
                        )

                    slides_to_render = branded
                except Exception:
                    slides_to_render = slides

            st.write(f"Channel: {str(channel_name or '').strip() or 'Brutally Honest Review'}")
            st.write(f"Title: {title_txt}")

            # For Shorts we keep a persistent title+stamp overlay throughout the video.
            pkg = None
            # Always generate a thumbnail for Shorts (no LLM unless YouTube metadata is enabled).
            if vt == "shorts" or youtube_metadata:
                try:
                    # Avoid LLM calls unless user explicitly asked for youtube metadata.
                    segs_for_pkg = segments if youtube_metadata else None
                    pkg = generate_youtube_package(
                        segs_for_pkg,
                        slides=slides,
                        topic=topic or None,
                        channel_name=channel_name,
                        title=title_txt,
                        video_type=vt,
                    )
                    idx = max(0, min(len(slides) - 1, int(pkg.thumbnail_slide_index))) if pkg else 0
                    bg = Path(slides[idx].image_path)
                    if not bg.is_absolute():
                        bg = (workspace / bg).resolve()

                    if vt == "shorts":
                        try:
                            from videogenerator.youtube import _try_find_raw_for_card

                            raw = _try_find_raw_for_card(bg)
                            if raw is not None:
                                bg = raw
                        except Exception:
                            pass
                    create_thumbnail(
                        out_path=out_dir / "thumbnail.png",
                        background_image=bg,
                        text=title_txt,
                        verdict_text=(pkg.verdict_label if pkg else None),
                        stamp_text=(pkg.thumbnail_stamp_text if pkg else None),
                        match_video_frame=False,
                        width=int(vid_w),
                        height=int(vid_h),
                        theme=("highlight" if vt == "shorts" else "default"),
                        show_title=(vt != "shorts"),
                    )
                except Exception:
                    pkg = None

            if vt == "shorts":
                try:
                    from videogenerator.youtube import overlay_shorts_title_and_stamp

                    slides_to_render = overlay_shorts_title_and_stamp(
                        slides_to_render,
                        out_dir=out_dir / "slides_overlay",
                        title=title_txt,
                        stamp_text=(pkg.thumbnail_stamp_text if pkg else None),
                        width=int(vid_w),
                        height=int(vid_h),
                           show_title=False,
                    )
                except Exception:
                    pass

            st.write("Rendering MP4…")
            out_mp4 = out_dir / "video.mp4"
            render_slideshow(
                slides_to_render,
                str(saved_audio),
                out_mp4,
                width=int(vid_w),
                height=int(vid_h),
                fps=30,
                bgm_path=None,
                bgm_volume=float(bgm_volume),
                bgm_duck=bool(bgm_duck),
                bgm_generate=False,
                bgm_preset=None if bgm_preset == "(none)" else bgm_preset,
                intro_seconds=float(intro_s),
                outro_seconds=float(outro_s),
                transition=None if run_transition == "none" else run_transition,
                transition_seconds=float(run_transition_seconds),
                ken_burns=False,
            )

            if vt == "review":
                st.write("Generating Shorts highlights…")
                try:
                    from videogenerator.review_highlights_shorts import make_shorts_from_review_highlights

                    shorts_dir = make_shorts_from_review_highlights(
                        review_audio_path=saved_audio,
                        review_out_dir=out_dir,
                        topic=(topic or None),
                        image_provider=image_provider,
                        serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
                        whisper_model="small",
                        min_image_width=900,
                        llm_model="gpt-4o-mini",
                        llm_pick_images=True,
                        reuse_images=bool(reuse_images),
                    )
                    if shorts_dir is not None:
                        st.success(f"Shorts created: {str(Path(shorts_dir) / 'video.mp4')}")
                    else:
                        st.caption("Shorts not created (missing transcript or no highlight clips).")
                except Exception:
                    st.caption("Shorts not created (requires OPENAI_API_KEY and a successful transcript).")

            if youtube_metadata:
                st.write("Generating YouTube metadata + thumbnail…")
                if pkg is None:
                    pkg = generate_youtube_package(
                        segments,
                        slides=slides,
                        topic=topic or None,
                        channel_name=channel_name,
                        title=title_txt,
                        video_type=vt,
                    )
                write_youtube_metadata_text(out_dir, pkg)

                st.write(
                    {
                        "thumbnail_slide_index": int(pkg.thumbnail_slide_index),
                        "verdict": pkg.verdict_label,
                        "stamp": pkg.thumbnail_stamp_text,
                    }
                )

                # Thumbnail already created above for Shorts; for non-shorts regenerate here.
                if vt != "shorts":
                    idx = max(0, min(len(slides) - 1, int(pkg.thumbnail_slide_index)))
                    bg = Path(slides[idx].image_path)
                    if not bg.is_absolute():
                        bg = (workspace / bg).resolve()

                    create_thumbnail(
                        out_path=out_dir / "thumbnail.png",
                        background_image=bg,
                        text=title_txt,
                        verdict_text=pkg.verdict_label,
                        stamp_text=pkg.thumbnail_stamp_text,
                        match_video_frame=False,
                    )

            if verify_video:
                st.write("Verifying MP4…")
                v = verify_local(out_mp4)
                st.write(v.duration_line.strip())
                st.write({"has_audio": v.has_audio, "audio_peak": v.audio_peak, "frame_hashes": v.frame_hashes})

            status.update(label="Done", state="complete", expanded=False)

        st.success("Finished")

    # Convenience links/actions
    if out_dir.exists():
        st.divider()
        st.subheader("Open outputs")
        st.write("Output folder:")
        st.code(str(out_dir), language="text")

        cols = st.columns([1, 1, 2])
        with cols[0]:
            if st.button("Open folder"):
                _open_folder(out_dir)
        with cols[1]:
            video_path = out_dir / "video.mp4"
            if video_path.exists():
                st.write("Video:")
                st.code(str(video_path), language="text")
        with cols[2]:
            thumb = out_dir / "thumbnail.png"
            if thumb.exists():
                st.image(str(thumb), caption="thumbnail.png", width="stretch")
