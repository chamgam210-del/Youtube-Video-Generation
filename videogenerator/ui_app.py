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
from videogenerator.render import bgm_preset_available, render_slideshow
from videogenerator.models import Slide
from videogenerator.verify_video import verify_local
from videogenerator.youtube import (
    YouTubePackage,
    create_thumbnail,
    generate_youtube_package,
    pick_review_thumbnail_text_with_llm,
    pick_long_review_thumbnail_with_vision,
    write_youtube_metadata_text,
)


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
    with st.expander("0) Generate Shorts scripts from text (GPT-5.2)", expanded=False):
        st.caption("Paste your raw review thoughts. The app generates 30–40s Shorts scripts with hook → tension → proof → payoff → loop.")

        shorts_text = st.text_area(
            "Review thoughts (text)",
            value="",
            height=220,
            placeholder="Paste your review thoughts here…",
        )
        shorts_count = st.selectbox("How many scripts?", options=[3, 4, 5], index=2)
        gen_scripts = st.button("Generate Shorts scripts", type="secondary")

        if gen_scripts:
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OPENAI_API_KEY is not set (add it to .env).")
            else:
                try:
                    from videogenerator.shorts_scripts import (
                        format_script_bracketed,
                        generate_shorts_scripts_from_review_text_with_llm,
                    )

                    scripts = generate_shorts_scripts_from_review_text_with_llm(
                        shorts_text,
                        topic=(st.session_state.get("image_subject") or st.session_state.get("topic_hint") or None),
                        count=int(shorts_count),
                        model="gpt-5.2",
                    )

                    for i, sc in enumerate(scripts, start=1):
                        st.markdown(f"**Short {i}: {sc.title}**")
                        st.caption(f"Thumbnail text: {sc.thumbnail_text}")
                        st.code(format_script_bracketed(sc), language="text")
                except Exception as e:
                    st.error(f"Failed to generate scripts: {e}")

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

    image_subject = st.text_input(
        "Visual subject for images (optional override)",
        key="image_subject",
        help="Force image search to stay on a specific movie/TV show (useful when your Shorts script never says the title).",
    )

    image_provider = st.selectbox(
        "Image provider",
        options=["google_images", "serpapi", "wikimedia"],
        index=0,
        help="google_images uses SerpAPI Google Images results directly (no license validation).",
    )

    video_type = st.selectbox(
        "Video type",
        options=["review (images only)", "explainer (text + images)", "commentary (clip insertion)", "shorts (9:16)", "shorts review (9:16, retention)", "auto"],
        index=0,
        help="Explainer/shorts use LLM-planned text-on-slide storyboards. Commentary auto-inserts referenced clips and compilations.",
    )

    # When creating Shorts, retention usually improves with faster cuts and more images.
    # We set dynamic defaults when the user switches video_type.
    last_vt = st.session_state.get("_last_video_type")
    if last_vt != video_type:
        if video_type.startswith("shorts"):
            st.session_state["max_images_slider"] = 20
            st.session_state["min_seg_seconds_slider"] = 1.8
            st.session_state["transition_sel"] = "none"
            st.session_state["transition_seconds_slider"] = 0.0
            # Shorts should start immediately; no intro/outro branding.
            st.session_state["intro_seconds_slider"] = 0.0
            st.session_state["outro_seconds_slider"] = 0.0
        else:
            st.session_state["max_images_slider"] = 12
            st.session_state["min_seg_seconds_slider"] = 6.0
            st.session_state["transition_sel"] = "fade"
            st.session_state["transition_seconds_slider"] = 0.35
            st.session_state["intro_seconds_slider"] = 2.5
            st.session_state["outro_seconds_slider"] = 3.0
        st.session_state["_last_video_type"] = video_type

    # Shorts Review auto-determines slide/beat count from audio duration.
    max_images: int | None = None
    if video_type != "shorts review (9:16, retention)":
        max_images = st.slider(
            "Max images",
            min_value=4,
            max_value=24,
            value=int(st.session_state.get("max_images_slider", 12)),
            step=1,
            key="max_images_slider",
        )
    min_seg_seconds = st.slider(
        "Min seconds per slide",
        min_value=1.0,
        max_value=12.0,
        value=float(st.session_state.get("min_seg_seconds_slider", 6.0)),
        step=0.1,
        key="min_seg_seconds_slider",
        help="For Shorts, 1.5–2.2s usually retains better than 6s holds.",
    )

    transition = st.selectbox(
        "Transition",
        options=["fade", "none"],
        index=0,
        key="transition_sel",
    )
    transition_seconds = st.slider(
        "Transition seconds",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("transition_seconds_slider", 0.35)),
        step=0.05,
        key="transition_seconds_slider",
    )

    st.subheader("3) Audio mix")
    bgm_preset = st.selectbox(
        "BGM preset",
        options=["elevator", "ambient", "creepy", "hiphop", "rnb", "clown", "cylinder_five", "dark_walk", "(none)"],
        index=0,
    )

    # Optional: upload a local BGM file and save it under the selected preset name.
    # This avoids manual file copying/renaming for file-backed presets like `cylinder_five`.
    bgm_upload = st.file_uploader(
        "Upload BGM file (optional)",
        type=["mp3", "wav", "m4a", "aac", "flac", "ogg"],
        help="If you upload a track while a file-backed preset is selected (e.g. cylinder_five), the UI saves it into assets/bgm/ so the preset can be used.",
    )

    if bgm_upload is not None and bgm_preset not in {"(none)", "", "ambient", "elevator", "creepy", "hiphop", "rnb", "clown"}:
        try:
            assets_bgm_dir = Path.cwd() / "assets" / "bgm"
            assets_bgm_dir.mkdir(parents=True, exist_ok=True)

            ext = Path(bgm_upload.name).suffix.lower()
            if ext not in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}:
                ext = ".mp3"

            target = assets_bgm_dir / f"{str(bgm_preset).strip().lower()}{ext}"
            data = bgm_upload.getvalue()

            # Only write if content changed.
            new_hash = hashlib.sha256(data).hexdigest()
            old_hash = None
            if target.exists():
                try:
                    old_hash = hashlib.sha256(target.read_bytes()).hexdigest()
                except Exception:
                    old_hash = None
            if new_hash != old_hash:
                target.write_bytes(data)

            st.success(f"Saved BGM preset file: {str(target)}")
        except Exception as e:
            st.warning(f"Failed to save BGM upload: {e}")

    if bgm_preset not in {"(none)", ""}:
        try:
            if not bgm_preset_available(str(bgm_preset)):
                st.warning(
                    "Selected BGM preset file not found. "
                    "Put it in assets/bgm/ (e.g. 'cylinder_five.mp3' or 'Cylinder Five - Chris Zabriskie.mp3') "
                    "or set VIDEOGENERATOR_BGM_DIR. "
                    "Continuing with no BGM."
                )
                bgm_preset = "(none)"
        except Exception:
            pass
    bgm_volume = st.slider("BGM volume", min_value=0.0, max_value=0.30, value=0.16, step=0.01)
    bgm_duck = st.checkbox("Ducking (reduce BGM under narration)", value=True)

    st.subheader("4) Branding")
    channel_name = st.text_input("Channel name", value="Brutally Honest Review")
    thumbnail_phrase_override = st.text_input(
        "Thumbnail phrase (review override, 2–4 words)",
        value="",
        help="If set, this exact phrase is used for the review thumbnail text (e.g. WORTH IT?, SURPRISINGLY GOOD).",
    )
    thumbnail_title_override = st.text_input(
        "Thumbnail title (review override)",
        value="",
        help="Optional: overrides the title text drawn on the long-review thumbnail. Leave blank to use the generated video title.",
    )
    thumbnail_stamp_override = st.text_input(
        "Thumbnail stamp (review override)",
        value="",
        help="Optional: overrides the verdict stamp text (e.g. GARBAGE!). Leave blank to use the generated verdict label.",
    )
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
    intro_seconds = st.slider(
        "Intro seconds",
        min_value=0.0,
        max_value=6.0,
        value=float(st.session_state.get("intro_seconds_slider", 2.5)),
        step=0.5,
        key="intro_seconds_slider",
    )
    outro_seconds = st.slider(
        "Outro seconds",
        min_value=0.0,
        max_value=8.0,
        value=float(st.session_state.get("outro_seconds_slider", 3.0)),
        step=0.5,
        key="outro_seconds_slider",
    )

    reuse_images = st.checkbox("Reuse images across reruns", value=True)
    fresh_images_this_run = st.checkbox(
        "Fetch fresh images this run",
        value=False,
        help="If enabled, this run will NOT reuse images from prior outputs (forces new downloads/selection). Useful for Review and Thumbnail-only rerolls.",
    )
    mix_video_clips = st.checkbox(
        "Mix in video clips (B-roll)",
        value=False,
        help="Use AI to find and splice relevant video clips (e.g. movie trailers, scenes) into the video as B-roll. Requires SERPAPI_API_KEY and yt-dlp.",
    )
    max_video_clips = 6
    if mix_video_clips:
        max_video_clips = st.slider(
            "Max video clips",
            min_value=1,
            max_value=12,
            value=6,
            step=1,
            help="Maximum number of video clips the AI can insert.",
        )
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

    b1, b2 = st.columns([1, 1])
    with b1:
        run_clicked = st.button("Run", type="primary")
    with b2:
        thumb_only_clicked = st.button("Thumbnail only", type="secondary", help="Generate thumbnail.png from the audio (transcribe → pick image + crop + text). Skips MP4 rendering.")

    action = "run" if run_clicked else ("thumbnail_only" if thumb_only_clicked else None)

    if action is not None:
        thumbnail_only = action == "thumbnail_only"
        out_dir.mkdir(parents=True, exist_ok=True)

        vt = "review"
        if video_type.startswith("explainer"):
            vt = "explainer"
        elif video_type.startswith("commentary"):
            vt = "commentary"
        elif video_type.startswith("shorts review"):
            vt = "shorts_review"
        elif video_type.startswith("shorts"):
            vt = "shorts"
        elif video_type == "auto":
            vt = "auto"

        # Render sizing preset.
        vid_w, vid_h = (1920, 1080)
        if vt in {"shorts", "shorts_review"}:
            vid_w, vid_h = (1080, 1920)

        # Shorts defaults: no transitions.
        run_max_images = int(max_images) if max_images is not None else None
        run_transition = transition
        run_transition_seconds = float(transition_seconds)
        if vt in {"shorts", "shorts_review"}:
            run_transition = "none"
            run_transition_seconds = 0.0

        # Pipeline
        with st.status("Running pipeline…", expanded=True) as status:
            st.write("Planning slides, searching images, writing timeline…")
            effective_reuse_images = bool(reuse_images) and (not bool(fresh_images_this_run))
            if fresh_images_this_run:
                st.caption("Fresh images enabled: not reusing prior output assets for this run.")
            run(
                audio_path=str(saved_audio),
                out_dir=out_dir,
                topic=(str(image_subject).strip() or str(topic).strip() or None),
                video_type=vt,
                image_provider=image_provider,
                serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
                max_images=(int(run_max_images) if run_max_images is not None else 12),
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
                reuse_images=bool(effective_reuse_images),
                mix_video_clips=bool(mix_video_clips),
                max_video_clips=int(max_video_clips),
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
                        if fresh_images_this_run:
                            st.caption("Did not reuse images (fresh images run).")
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

            # Shorts: no channel intro/outro branding slates.
            if vt == "shorts":
                intro_s = 0.0
                outro_s = 0.0

            # Shorts Review: audio starts after the hook frame (hook is produced by the pipeline timeline).
            if vt == "shorts_review":
                intro_s = 1.6
                outro_s = 0.0

            if (vt not in {"shorts", "shorts_review"}) and (intro_s > 0.0 or outro_s > 0.0):
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
                                motion=getattr(s, "motion", None),
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
            # Always generate a thumbnail for Shorts/Reviews, or when user asked for metadata,
            # or when the user chose "Thumbnail only".
            # Shorts Review: skip thumbnail asset generation (captions/keywords only workflow).
            if vt in {"shorts", "review"} or youtube_metadata or thumbnail_only:
                try:
                    # Avoid LLM calls unless user explicitly asked for youtube metadata.
                    segs_for_pkg = segments if (youtube_metadata or thumbnail_only) else None
                    pkg = generate_youtube_package(
                        segs_for_pkg,
                        slides=slides,
                        topic=topic or None,
                        channel_name=channel_name,
                        title=title_txt,
                        video_type=vt,
                    )

                    if vt == "review":
                        try:
                            forced_phrase = (thumbnail_phrase_override or "").strip() or None
                            if forced_phrase is None:
                                try:
                                    forced_phrase = pick_review_thumbnail_text_with_llm(
                                        segments,
                                        title=title_txt,
                                        model="gpt-4o-mini",
                                    )
                                except Exception:
                                    forced_phrase = None
                            idx_v, thumb_text, thumb_crop = pick_long_review_thumbnail_with_vision(
                                slides=slides,
                                topic=topic or None,
                                title=title_txt,
                                model="gpt-4o-mini",
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

                    idx = max(0, min(len(slides) - 1, int(pkg.thumbnail_slide_index))) if pkg else 0
                    bg = Path(slides[idx].image_path)
                    if not bg.is_absolute():
                        bg = (workspace / bg).resolve()

                    if vt in {"shorts", "shorts_review"}:
                        try:
                            from videogenerator.youtube import _try_find_raw_for_card

                            raw = _try_find_raw_for_card(bg)
                            if raw is not None:
                                bg = raw
                        except Exception:
                            pass
                    # Long review thumbnail overrides (title + stamp).
                    thumb_title = (thumbnail_title_override or "").strip()
                    if not thumb_title:
                        try:
                            thumb_title = str(pkg.title or title_txt).strip() if pkg else str(title_txt).strip()
                        except Exception:
                            thumb_title = str(title_txt).strip()

                    stamp_raw = (thumbnail_stamp_override or "").strip()
                    if stamp_raw.lower() in {"(none)", "none", "off", "disabled"}:
                        stamp_raw = ""
                    thumb_stamp = stamp_raw
                    if not thumb_stamp:
                        thumb_stamp = str((pkg.verdict_label if pkg else "") or "").strip() or None

                    create_thumbnail(
                        out_path=out_dir / "thumbnail.png",
                        background_image=bg,
                        text=(thumb_title if vt == "review" else title_txt),
                        verdict_text=(thumb_stamp if vt == "review" else (pkg.verdict_label if pkg else None)),
                        stamp_text=(None if vt == "review" else (pkg.thumbnail_stamp_text if pkg else None)),
                        match_video_frame=False,
                        width=int(vid_w),
                        height=int(vid_h),
                        theme=("highlight" if vt in {"shorts"} else ("review_long" if vt == "review" else "default")),
                        show_title=(True if vt == "review" else (vt not in {"shorts"})),
                        crop=(pkg.thumbnail_crop if (vt == "review" and pkg is not None) else None),
                    )
                except Exception:
                    pkg = None

            if vt in {"shorts"}:
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

            if thumbnail_only:
                status.update(label="Thumbnail ready", state="complete", expanded=False)
                st.success("Thumbnail generated")
                st.stop()

            st.write("Rendering MP4…")
            out_mp4 = out_dir / "video.mp4"

            # Shorts Review may produce a padded narration track (with pivot silences).
            audio_for_render: Path = saved_audio
            try:
                meta_path = out_dir / "run_meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    rp = (meta or {}).get("render_audio_path")
                    if rp:
                        p = Path(str(rp))
                        if p.exists() and p.stat().st_size > 4096:
                            audio_for_render = p
            except Exception:
                pass

            effective_bgm_preset = None if bgm_preset == "(none)" else bgm_preset
            if vt == "review" and effective_bgm_preset is None:
                try:
                    from videogenerator.youtube import _fallback_verdict_label

                    v = ""
                    if pkg is not None:
                        v = str(pkg.verdict_label or "").strip()
                    if not v:
                        v = _fallback_verdict_label(segments=segments)
                    if v == "Garbage!":
                        effective_bgm_preset = "clown"
                except Exception:
                    pass

            render_slideshow(
                slides_to_render,
                str(audio_for_render),
                out_mp4,
                width=int(vid_w),
                height=int(vid_h),
                fps=30,
                bgm_path=None,
                bgm_volume=float(bgm_volume),
                bgm_duck=bool(bgm_duck),
                bgm_generate=False,
                bgm_preset=effective_bgm_preset,
                intro_seconds=float(intro_s),
                outro_seconds=float(outro_s),
                transition=None if run_transition == "none" else run_transition,
                transition_seconds=float(run_transition_seconds),
                ken_burns=(vt == "shorts_review"),
            )

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

                if vt == "review":
                    try:
                        forced_phrase = (thumbnail_phrase_override or "").strip() or None
                        if forced_phrase is None:
                            try:
                                forced_phrase = pick_review_thumbnail_text_with_llm(
                                    segments,
                                    title=title_txt,
                                    model="gpt-4o-mini",
                                )
                            except Exception:
                                forced_phrase = None
                        idx_v, thumb_text, thumb_crop = pick_long_review_thumbnail_with_vision(
                            slides=slides,
                            topic=topic or None,
                            title=title_txt,
                            model="gpt-4o-mini",
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

                st.write(
                    {
                        "thumbnail_slide_index": int(pkg.thumbnail_slide_index),
                        "verdict": pkg.verdict_label,
                        "stamp": pkg.thumbnail_stamp_text,
                        "thumbnail_text": getattr(pkg, "thumbnail_text", None),
                        "thumbnail_crop": getattr(pkg, "thumbnail_crop", None),
                    }
                )

                # Thumbnail already created above for Shorts; for non-shorts regenerate here.
                if vt != "shorts":
                    idx = max(0, min(len(slides) - 1, int(pkg.thumbnail_slide_index)))
                    bg = Path(slides[idx].image_path)
                    if not bg.is_absolute():
                        bg = (workspace / bg).resolve()

                    # Long review thumbnail overrides (title + stamp) for the metadata regeneration path.
                    thumb_title = (thumbnail_title_override or "").strip()
                    if not thumb_title:
                        try:
                            thumb_title = str(pkg.title or title_txt).strip() or str(title_txt).strip()
                        except Exception:
                            thumb_title = str(title_txt).strip()

                    stamp_raw = (thumbnail_stamp_override or "").strip()
                    if stamp_raw.lower() in {"(none)", "none", "off", "disabled"}:
                        stamp_raw = ""
                    thumb_stamp = stamp_raw
                    if not thumb_stamp:
                        try:
                            thumb_stamp = str((pkg.verdict_label or "") if pkg else "").strip() or None
                        except Exception:
                            thumb_stamp = None

                    create_thumbnail(
                        out_path=out_dir / "thumbnail.png",
                        background_image=bg,
                        text=(thumb_title if vt == "review" else title_txt),
                        verdict_text=(thumb_stamp if vt == "review" else pkg.verdict_label),
                        stamp_text=(None if vt == "review" else pkg.thumbnail_stamp_text),
                        match_video_frame=False,
                        theme=("review_long" if vt == "review" else "default"),
                        show_title=(True if vt == "review" else True),
                        crop=(pkg.thumbnail_crop if vt == "review" else None),
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

        # Players (render inside columns so they don't take the full page width)
        video_path = out_dir / "video.mp4"
        if video_path.exists():
            left, _right = st.columns([2, 1])
            if video_path.exists():
                with left:
                    st.subheader("Full review")
                    try:
                        st.video(str(video_path))
                    except Exception:
                        st.video(video_path.read_bytes(), format="video/mp4")

        cols = st.columns([1, 1, 2])
        with cols[0]:
            if st.button("Open folder"):
                _open_folder(out_dir)
        with cols[1]:
            if video_path.exists():
                st.write("Video:")
                st.code(str(video_path), language="text")
        with cols[2]:
            thumb = out_dir / "thumbnail.png"
            if thumb.exists():
                st.image(str(thumb), caption="thumbnail.png", width="stretch")
