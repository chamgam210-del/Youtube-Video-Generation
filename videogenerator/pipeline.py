from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import time

from tqdm import tqdm

from .audio import get_audio_duration_seconds
from .segments import bucketize_segments, merge_short_segments, select_evenly_spaced
from .transcribe import transcribe_cached, write_transcript_files
from .models import Slide
from .utils import ensure_dir, extract_keywords, read_json, write_json
from .wikimedia import download_image, search_commons_image, search_commons_images


def run(
    *,
    audio_path: str | Path,
    out_dir: str | Path,
    topic: str | None = None,
    video_type: str = "review",  # review|explainer|shorts|shorts_review|auto
    image_provider: str = "wikimedia",
    serpapi_api_key: str | None = None,
    max_images: int = 12,
    min_images: int = 4,
    min_seg_seconds: float = 6.0,
    max_slide_seconds: float = 0.0,  # 0 = no limit
    whisper_model: str = "small",
    min_image_width: int = 900,
    video_width: int = 1920,
    video_height: int = 1080,
    cache_transcript: bool = True,
    cache_dir: str | Path | None = None,
    storyboard: str = "auto",  # auto|llm|heuristic
    llm_model: str = "gpt-4o-mini",
    llm_pick_images: bool = True,
    reuse_images: bool = True,
    pool_image_paths: list[str | Path] | None = None,
    mix_video_clips: bool = False,
    max_video_clips: int = 6,
    clip_queries: list[str] | None = None,
    clip_research: bool = False,
) -> Path:
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)

    audio_stat = audio_path.stat()

    assets_dir = ensure_dir(out_dir / "assets")

    audio_duration = get_audio_duration_seconds(audio_path)

    # NOTE: hook/ending durations depend on the FINAL video type (vt). We initialize
    # them here and set them after `vt` is resolved (including `auto`).
    hook_s = 0.0
    ending_s = 0.0
    timeline_duration = float(audio_duration)

    # Shorts Review pivot pauses: we may insert mid-track silence and shift the VIDEO timeline.
    pause_schedule: list[tuple[float, float]] = []  # (pause_start_video_time, pause_seconds)
    audio_pause_insertions: list[tuple[float, float]] = []  # (pause_at_audio_time, pause_seconds)
    render_audio_override: Path | None = None

    def _ensure_audio_with_pauses(*, audio_in: Path, insertions: list[tuple[float, float]]) -> Path | None:
        """Return a WAV audio path with inserted silence, or None if no-op.

        `insertions` times are in ORIGINAL audio time.
        """

        ins = [(max(0.0, float(t)), max(0.0, float(d))) for t, d in (insertions or []) if float(d) > 1e-6]
        if not ins:
            return None

        # De-dup (time, duration) pairs; keep deterministic order.
        seen: set[tuple[float, float]] = set()
        uniq: list[tuple[float, float]] = []
        for t, d in ins:
            key = (round(t, 3), round(d, 3))
            if key in seen:
                continue
            seen.add(key)
            uniq.append((t, d))
        uniq.sort(key=lambda x: x[0])

        out_wav = assets_dir / "narration_with_pivot_pauses.wav"

        # Build a fingerprint of the insertion parameters so we can detect
        # when a cached WAV was produced with DIFFERENT insertions (e.g.
        # a previous run that used a different cue-phrase timestamp).
        import hashlib as _hl
        ins_fingerprint = _hl.md5(
            "|".join(f"{t:.3f},{d:.3f}" for t, d in uniq).encode()
        ).hexdigest()[:12]
        fingerprint_file = out_wav.with_suffix(".fingerprint")

        try:
            if out_wav.exists() and out_wav.stat().st_size > 4096:
                if out_wav.stat().st_mtime >= audio_in.stat().st_mtime:
                    # Also verify that insertions haven't changed.
                    if fingerprint_file.exists() and fingerprint_file.read_text().strip() == ins_fingerprint:
                        return out_wav
        except Exception:
            pass

        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

        # Build: [seg0][sil0][seg1][sil1]...[segN] concat.
        # We trim from the ORIGINAL audio; inserted silence does not affect trim times.
        seg_labels: list[str] = []
        filters: list[str] = []

        prev = 0.0
        for i, (t, d) in enumerate(uniq):
            t = max(prev, float(t))
            # Audio segment up to insertion.
            filters.append(
                f"[0:a]atrim=start={prev:.6f}:end={t:.6f},asetpts=PTS-STARTPTS[a{i}]"
            )
            seg_labels.append(f"[a{i}]")

            # Insert silence.
            filters.append(
                f"anullsrc=r=44100:cl=stereo,atrim=0:{float(d):.6f},asetpts=PTS-STARTPTS[s{i}]"
            )
            seg_labels.append(f"[s{i}]")
            prev = t

        # Tail segment.
        filters.append(f"[0:a]atrim=start={prev:.6f},asetpts=PTS-STARTPTS[atail]")
        seg_labels.append("[atail]")

        concat_n = len(seg_labels)
        filter_complex = ";".join(filters) + ";" + "".join(seg_labels) + f"concat=n={concat_n}:v=0:a=1[outa]"

        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_in),
            "-filter_complex",
            filter_complex,
            "-map",
            "[outa]",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception:
            return None

        # Write fingerprint so we can detect stale caches.
        try:
            fingerprint_file.write_text(ins_fingerprint)
        except Exception:
            pass

        return out_wav

    # Default transcript cache should be stable across output folders.
    effective_cache_dir = Path(cache_dir) if cache_dir else (Path.cwd() / ".cache" / "transcripts")
    segments = transcribe_cached(
        audio_path,
        model_name=whisper_model,
        cache_dir=effective_cache_dir,
        use_cache=cache_transcript,
    )

    # Always write transcript files into the output folder for convenience.
    try:
        write_transcript_files(segments, out_dir=out_dir)
    except Exception:
        pass

    # If whisper produced a 0-length fallback segment, expand to full duration
    if len(segments) == 1 and segments[0].start == 0.0 and segments[0].end == 0.0:
        segments = [segments[0].__class__(start=0.0, end=audio_duration, text=segments[0].text)]

    vt = (video_type or "review").strip().lower()
    if vt not in {"review", "explainer", "shorts", "shorts_review", "commentary", "clip_review", "auto"}:
        vt = "review"

    # Shorts Review determines its own beat count from audio duration.
    # (We keep `max_images` for other video types.)
    shorts_review_max_beats: int | None = None

    def _shorts_review_target_total_images(dur_s: float) -> int:
        """Target total DISTINCT image windows for a shorts_review.

        Spec:
        - 30–40s Short => 6–8 images max
        """

        try:
            d = float(dur_s)
        except Exception:
            d = 0.0

        if d <= 0.0:
            return 6

        # Rule-of-thumb: ~1 new idea every ~5s, clamped to a small set.
        # For 30–40s: round(d/5) => 6..8.
        import math

        approx = int(round(d / 5.0))
        if 30.0 <= d <= 40.0:
            return int(min(8, max(6, approx)))

        # Outside that range, keep shorts_review tight anyway.
        return int(min(8, max(4, approx)))

    # Shorts Review is a retention-first format: enforce fast beat pacing by building near-uniform
    # transcript buckets (~1-2s). This gives the storyboard and image picker many cut points.
    if vt == "shorts_review":
        # Calm the cadence: one image = one idea.
        # Minimum hold: 2.0s. Sweet spot: ~2.5–3.5s.
        target_s = 3.0
        merged = bucketize_segments(
            segments,
            target_seconds=target_s,
            min_seconds=2.0,
            max_seconds=3.6,
        )
        # Compute beat count from audio duration (ignore user-provided max_images).
        # Keep it tight: total images (incl. hook + ending) should be small.
        total_images_target = _shorts_review_target_total_images(float(audio_duration))
        shorts_review_max_beats = max(1, int(total_images_target) - 2)
    else:
        merged = merge_short_segments(segments, min_seconds=min_seg_seconds)

    use_llm = storyboard in {"llm", "auto"}

    # For non-review videos, we want a stable topic hint to keep image searches on the right subject.
    effective_topic: str | None = (topic or "").strip() or None
    topic_type: str | None = None
    # For long-form reviews, treat image topic inference like shorts/commentary
    # so that image search benefits from the same LLM “agentic” planning.
    if vt in {"review", "explainer", "shorts", "shorts_review", "commentary", "clip_review", "auto"} and not effective_topic:
        if use_llm:
            try:
                from .llm_storyboard import infer_topic_with_llm

                inferred = infer_topic_with_llm(merged, model=llm_model)
                effective_topic = inferred.topic.strip() or None
                topic_type = inferred.topic_type
            except Exception:
                effective_topic = None
        if not effective_topic:
            from .llm_storyboard import infer_topic_fallback

            inferred = infer_topic_fallback(audio_stem=audio_path.stem)
            effective_topic = inferred.topic.strip() or None
            topic_type = inferred.topic_type

    # Heuristic: if transcript strongly suggests TV content, treat as tv_show.
    if vt in {"review", "explainer", "shorts", "shorts_review", "commentary", "clip_review"} and (topic_type is None or topic_type == "other"):
        try:
            tail = " ".join(s.text for s in merged[-25:]).lower()
        except Exception:
            tail = ""
        tv_hints = (
            "tv",
            "tv show",
            "series",
            "episode",
            "season",
            "cast",
            "character",
            "apple tv",
            "netflix",
            "hbo",
        )
        if any(h in tail for h in tv_hints):
            topic_type = "tv_show"

    # planned items: (start, end, query, headline, subhead)
    planned: list[tuple[float, float, str, str, str | None]] = []
    planned_motion: list[str | None] | None = None

    if vt == "auto":
        if use_llm:
            try:
                from .llm_storyboard import classify_transcript_kind_with_llm

                vt = classify_transcript_kind_with_llm(merged, topic=topic, model=llm_model)
            except Exception:
                from .llm_storyboard import classify_transcript_kind_fallback

                vt = classify_transcript_kind_fallback(merged)
        else:
            from .llm_storyboard import classify_transcript_kind_fallback

            vt = classify_transcript_kind_fallback(merged)

    # Shorts Review retention: hold a longer hook frame to let the topic land.
    if vt == "shorts_review":
        # Spec: hook image >= 3.0s.
        hook_s = 3.0
        # Keep CTA readable; default to a calm 3s hold.
        ending_s = 3.0
        # Video must end when the (hook-delayed) audio ends.
        # CTA lives inside the last `ending_s` seconds of this timeline.
        timeline_duration = float(audio_duration) + float(hook_s)
    else:
        hook_s = 0.0
        ending_s = 0.0
        timeline_duration = float(audio_duration)

    if use_llm:
        try:
            if vt == "shorts_review":
                from .llm_storyboard import plan_shorts_review_storyboard_with_llm, _parse_timestamp_range

                sb = plan_shorts_review_storyboard_with_llm(
                    merged,
                    audio_duration=audio_duration,
                    topic=effective_topic,
                    max_beats=int(shorts_review_max_beats or 12),
                    hook_seconds=float(hook_s),
                    ending_seconds=float(ending_s),
                    model=llm_model,
                )

                def _q_for_image_type(image_type: str, *, topic_name: str | None, hook_type: str | None = None) -> str:
                    t = (topic_name or "").strip()
                    it = (image_type or "").strip().lower()
                    if not t:
                        return ""
                    if it == "poster":
                        return f"{t} dark moody official poster"
                    if it in {"closeup", "reaction_closeup"}:
                        # Bias brighter/closer to create a strong reset vs the hook poster.
                        return f"{t} brighter close up still"
                    if it == "prestige_still":
                        return f"{t} cinematic scene still"
                    if it == "neutral" and (hook_type or "").strip().lower() == "poster":
                        return f"{t} bright close up still"
                    return f"{t} scene still"

                def _motion_map(m: str) -> str:
                    mm = (m or "").strip().lower()
                    if mm in {"none", "minimal"}:
                        return "hold"
                    if mm == "snap_zoom":
                        return "snap"
                    # slow_zoom / slow_zoom_in
                    return "zoom_in"

                # Hook frame planned at [0, hook_s].
                planned = [
                    (
                        0.0,
                        float(hook_s),
                        _q_for_image_type(sb.hook_frame.image_type, topic_name=effective_topic, hook_type=sb.hook_frame.image_type),
                        str(sb.hook_frame.text or "").strip(),
                        None,
                    )
                ]
                planned_motion = ["hold"]

                # Beats (already in VIDEO time, shifted by hook_s).
                pause_s = 0.3
                time_shift = 0.0

                # Choose a single pivot beat (max 1). Prefer 30–60% into the short.
                pivot_candidates: list[tuple[int, float]] = []
                for bi, b in enumerate(sb.beats or []):
                    rng = _parse_timestamp_range(getattr(b, "timestamp", ""))
                    if not rng:
                        continue
                    st0 = float(rng[0])
                    low_line0 = str(getattr(b, "line", "") or "").lower()
                    is_pivot0 = (
                        (" but " in f" {low_line0} ")
                        or ("however" in low_line0)
                        or ("here's the thing" in low_line0)
                        or ("heres the thing" in low_line0)
                        or ("the problem is" in low_line0)
                    )
                    if is_pivot0:
                        pivot_candidates.append((int(bi), float(st0)))

                pivot_idx: int | None = None
                if pivot_candidates and audio_duration > 0:
                    target = float(hook_s) + float(audio_duration) * 0.45
                    lo = float(hook_s) + max(5.0, float(audio_duration) * 0.30)
                    hi = float(hook_s) + max(5.0, float(audio_duration) * 0.60)
                    in_range = [(bi, st0) for bi, st0 in pivot_candidates if (float(lo) <= float(st0) <= float(hi))]
                    pool = in_range if in_range else pivot_candidates
                    pool.sort(key=lambda x: abs(float(x[1]) - float(target)))
                    pivot_idx = int(pool[0][0]) if pool else None

                for bi, b in enumerate(sb.beats):
                    rng = _parse_timestamp_range(b.timestamp)
                    if not rng:
                        continue
                    st, en = float(rng[0]), float(rng[1])
                    if en <= st:
                        continue

                    # Apply any previous pivot pauses.
                    st_v = float(st) + float(time_shift)
                    en_v = float(en) + float(time_shift)

                    # Pivot pause (Rule B): insert a real 0.3s silence + snap before the pivot words.
                    low_line = str(b.line or "").lower()
                    is_pivot = (
                        (" but " in f" {low_line} ")
                        or ("however" in low_line)
                        or ("here's the thing" in low_line)
                        or ("heres the thing" in low_line)
                        or ("the problem is" in low_line)
                    )
                    if is_pivot and (pivot_idx is not None) and (int(bi) == int(pivot_idx)):
                        # Insert silence at the *audio* time where this beat begins (undo hook offset).
                        pause_at_audio = max(0.0, float(st) - float(hook_s))
                        audio_pause_insertions.append((pause_at_audio, float(pause_s)))
                        pause_schedule.append((float(st_v), float(pause_s)))

                        q_pivot = _q_for_image_type("closeup", topic_name=effective_topic, hook_type=sb.hook_frame.image_type)
                        planned.append((float(st_v), float(st_v + pause_s), q_pivot, "BUT…", None))
                        planned_motion.append("snap")

                        # Shift the spoken portion forward.
                        time_shift += float(pause_s)
                        st_v = float(st_v) + float(pause_s)
                        en_v = float(en_v) + float(pause_s)

                    # Keep the LLM beat window intact. (Avoid double-splitting that creates
                    # micro-beats and a jittery feel.)
                    headline = str(b.text or "").strip()
                    motion = _motion_map(b.motion)
                    q = _q_for_image_type(b.image_type, topic_name=effective_topic, hook_type=sb.hook_frame.image_type)
                    planned.append((float(st_v), float(en_v), q, headline, None))
                    planned_motion.append(motion)

                # Update timeline duration to include inserted pivot silences.
                # (CTA lives inside the last `ending_s` seconds; do not extend the timeline.)
                timeline_duration = float(audio_duration) + float(hook_s) + float(time_shift)

                # Ending frame: reserve last ending_s seconds as a static loop card.
                if ending_s > 0 and timeline_duration > ending_s:
                    end_start = max(float(hook_s), float(timeline_duration) - float(ending_s))
                    planned.append(
                        (
                            float(end_start),
                            float(timeline_duration),
                            _q_for_image_type(sb.ending_frame.image_type, topic_name=effective_topic, hook_type=sb.hook_frame.image_type) or (
                                f"{effective_topic} official poster".strip() if effective_topic else ""
                            ),
                            (str(sb.ending_frame.text or "").strip() or "AGREE? 👇"),
                            None,
                        )
                    )
                    planned_motion.append("hold")

                # Sort and clamp.
                zipped = list(zip(planned, planned_motion or []))
                zipped.sort(key=lambda x: float(x[0][0]))
                planned = [p for p, _m in zipped]
                planned_motion = [_m for _p, _m in zipped]

                # Caption timing clamp (Rule C): prevent accidental lingering by inserting explicit
                # no-text filler windows for any gaps.
                if planned and planned_motion:
                    clamped: list[tuple[float, float, str, str, str | None]] = []
                    clamped_motion: list[str | None] = []
                    for i, (st, en, q, h, sh) in enumerate(planned):
                        clamped.append((float(st), float(en), q, h, sh))
                        clamped_motion.append(planned_motion[i])

                        if i + 1 < len(planned):
                            nxt_st = float(planned[i + 1][0])
                            gap = float(nxt_st) - float(en)
                            if gap > 1e-3:
                                # Continue the same background without text.
                                clamped.append((float(en), float(nxt_st), q, "", None))
                                clamped_motion.append("hold")

                    planned = clamped
                    planned_motion = clamped_motion

            elif vt in {"explainer", "shorts"}:
                from .llm_storyboard import plan_rich_slides_with_llm

                rich = plan_rich_slides_with_llm(
                    merged,
                    audio_duration=audio_duration,
                    topic=effective_topic,
                    max_slides=max_images,
                    kind=("shorts" if vt in {"shorts"} else "explainer"),
                    model=llm_model,
                )
                planned = [(s.start, s.end, s.query, s.headline, s.subhead) for s in rich]
            else:
                from .llm_storyboard import plan_slides_with_llm

                # Give the LLM a condensed transcript (merged segments) so it can pick good cut points.
                story = plan_slides_with_llm(
                    merged,
                    audio_duration=audio_duration,
                    topic=effective_topic,
                    max_images=max_images,
                    model=llm_model,
                )
                planned = [(s.start, s.end, s.query, "", None) for s in story]

            # Guardrail: if the LLM plan front-loads slide changes and leaves a very long final hold,
            # re-space the slide boundaries across the full audio duration while keeping the LLM's
            # topic-ordered queries/headlines. This avoids the "hits max images then freezes" behavior.
            if planned and audio_duration > 0 and vt != "shorts_review":
                n = len(planned)
                avg = float(audio_duration) / float(n)
                last_start = float(planned[-1][0])
                last_dur = float(audio_duration) - last_start
                if last_dur > (1.75 * avg):
                    buckets = select_evenly_spaced(merged, max_items=n, audio_duration=audio_duration)
                    respaced: list[tuple[float, float, str, str, str | None]] = []
                    for i, b in enumerate(buckets):
                        q, h, sh = planned[i][2], planned[i][3], planned[i][4]
                        respaced.append((float(b.start), float(b.end), q, h, sh))
                    planned = respaced

            # Guardrail: if the LLM produced FAR fewer slides than requested
            # (e.g. 3 for a 5-minute audio when max_images=12), expand by evenly
            # distributing heuristic buckets and cycling the LLM queries.
            # We clear the cycled query for expanded slides so they rely on
            # per-slide transcript keywords + visual suffix rotation (diversity).
            if planned and audio_duration > 0 and vt not in {"shorts_review"}:
                _min_slides = max(int(min_images), max(4, int(audio_duration / 60)))  # at least 1 per minute
                target = max(_min_slides, min(max_images, int(audio_duration / 30)))
                if len(planned) < _min_slides:
                    print(f"[pipeline] LLM produced only {len(planned)} slides for {audio_duration:.0f}s — expanding to {target}")
                    buckets = select_evenly_spaced(merged, max_items=target, audio_duration=audio_duration)
                    expanded: list[tuple[float, float, str, str, str | None]] = []
                    for i, b in enumerate(buckets):
                        src = planned[i % len(planned)]
                        # Clear the LLM query for slides beyond the original set
                        # so they don't all repeat the same search query.
                        q = src[2] if i < len(planned) else ""
                        expanded.append((float(b.start), float(b.end), q, src[3], src[4]))
                    planned = expanded
        except Exception:
            if storyboard == "llm":
                raise
            planned = []

    if not planned and storyboard == "llm":
        raise RuntimeError("LLM storyboard produced no slides")

    if not planned:
        picked = select_evenly_spaced(
            merged,
            max_items=(int(shorts_review_max_beats or 12) if vt == "shorts_review" else max_images),
            audio_duration=audio_duration,
        )
        planned = [(s.start, s.end, "", "", None) for s in picked]

    # ── Guardrail: enforce min_seg_seconds as a hard floor on slide durations ──
    # Merge any planned slide shorter than the user's minimum into its neighbor.
    if planned and vt not in {"shorts_review"} and min_seg_seconds > 0:
        enforced: list[tuple[float, float, str, str, str | None]] = []
        for st, en, q, h, sh in planned:
            dur = float(en) - float(st)
            if dur < float(min_seg_seconds) and enforced:
                # Absorb into the previous slide (extend its end time).
                prev = enforced[-1]
                enforced[-1] = (prev[0], float(en), prev[2], prev[3], prev[4])
            else:
                enforced.append((float(st), float(en), q, h, sh))
        if enforced:
            planned = enforced

    # ── Guardrail: enforce max_slide_seconds — split any slide longer than the cap ──
    if planned and max_slide_seconds > 0 and vt not in {"shorts_review"}:
        split_planned: list[tuple[float, float, str, str, str | None]] = []
        for st, en, q, h, sh in planned:
            dur = float(en) - float(st)
            if dur > float(max_slide_seconds) + 0.5:
                # Split into chunks of max_slide_seconds.
                cursor = float(st)
                while cursor < float(en) - 0.5:
                    chunk_end = min(cursor + float(max_slide_seconds), float(en))
                    split_planned.append((cursor, chunk_end, q, h, sh))
                    cursor = chunk_end
            else:
                split_planned.append((float(st), float(en), q, h, sh))
        if split_planned:
            planned = split_planned
            print(f"[pipeline] max_slide_seconds={max_slide_seconds:.1f}s → {len(planned)} slides after splitting")

    # Shorts Review heuristic fallback: strictly keep a fast cadence by respacing the plan.
    # (LLM shorts_review schema path above bypasses this.)
    if vt == "shorts_review" and merged and audio_duration > 0 and not planned_motion:
        def _shorts_caption(text: str) -> str:
            t = (text or "").strip()
            low = t.lower()
            if not t:
                return ""

            # Direct triggers that should be big + simple.
            if "wild" in low:
                return "WILD"
            if low.startswith("but ") or low.startswith("but,") or low.startswith("but"):
                return "BUT…"
            if "entertaining" in low:
                return "ENTERTAINING"
            if "not bad" in low:
                return "NOT BAD"
            if "standards" in low and "changed" in low:
                return "STANDARDS CHANGED"
            if "agree" in low or t.endswith("?"):
                return "AGREE? 👇"

            # Otherwise, make an uppercase keyword-ish caption.
            kws = extract_keywords(t, max_words=6)
            if kws:
                return " ".join([k.upper() for k in kws[:4]])

            # Fallback: trimmed sentence fragment.
            return (t[:42] + "…") if len(t) > 45 else t

        def _shorts_motion(text: str, idx: int) -> str:
            t = (text or "").strip()
            low = t.lower()
            if ("agree" in low) or t.endswith("?"):
                return "hold"
            if low.startswith("but"):
                return "snap"
            if "wild" in low:
                return "snap"
            if idx == 0:
                return "hold"
            cycle = ["zoom_in", "pan_lr", "zoom_out", "pan_rl"]
            return cycle[(idx - 1) % len(cycle)]

        def _range_text(start: float, end: float) -> str:
            txt = ""
            for s in merged:
                if s.end <= start:
                    continue
                if s.start >= end:
                    break
                txt += " " + str(s.text or "")
            return txt.strip()

        # RULE 2: Non-uniform timing (fast early, variable middle, slower payoff, static end).
        pattern = [1.2, 1.3, 1.5, 1.8, 1.4, 1.2]
        total = float(audio_duration)
        windows: list[tuple[float, float]] = []
        t = 0.0
        pi = 0
        while t < total - 1e-6 and len(windows) < int(shorts_review_max_beats or 12):
            dur = float(pattern[pi % len(pattern)])
            # Early: slightly faster.
            if t <= (0.18 * total):
                dur = max(1.15, dur - 0.10)
            # Near end: keep the loop card clean.
            if total - t <= 1.35:
                dur = min(dur, total - t)
            end = min(total, t + dur)
            windows.append((float(t), float(end)))
            t = float(end)
            pi += 1

        # If we have too many windows (rare), trim by merging from the middle.
        if len(windows) > int(shorts_review_max_beats or 12):
            windows = windows[: int(shorts_review_max_beats or 12)]
            windows[-1] = (windows[-1][0], float(total))

        # Map any LLM content onto our timing windows by index.
        content = [(p[2], p[3], p[4]) for p in planned]
        respaced: list[tuple[float, float, str, str, str | None]] = []
        motions: list[str | None] = []

        for i, (st, en) in enumerate(windows):
            q = content[i][0] if i < len(content) else ""
            h = content[i][1] if i < len(content) else ""
            sh = content[i][2] if i < len(content) else None

            wtxt = _range_text(float(st), float(en))
            low = wtxt.lower()

            # RULE 3: Emphasis triggers.
            if ("but" in low) and ("here" in low or len(wtxt.split()) <= 6):
                h = "BUT…"
                m = "snap"
                if effective_topic:
                    q = f"{effective_topic} intense close up still".strip()
            elif "entertaining" in low:
                h = "ENTERTAINING"
                m = "zoom_in"
                if effective_topic:
                    q = f"{effective_topic} best still portrait".strip()
            else:
                if not str(h or "").strip():
                    h = _shorts_caption(wtxt)
                m = _shorts_motion(wtxt, i)

            # RULE 1: keep first narration window poster-like for preview tone continuity.
            if i == 0 and effective_topic:
                q = f"{effective_topic} official poster".strip()

            # RULE 4: final frame should be static and loop-friendly.
            if i == len(windows) - 1:
                h = "AGREE? 👇"
                m = "hold"
                # Keep tone compatible with the hook.
                if effective_topic:
                    q = f"{effective_topic} official poster".strip()
                # Clamp last duration to ~1.2s when possible.
                if (float(en) - float(st)) > 1.25 and total - float(st) >= 1.05:
                    en = float(st) + 1.20

            respaced.append((float(st), float(en), str(q or ""), str(h or "").strip(), (str(sh).strip() if sh else None)))
            motions.append(m)

        # Ensure we end exactly at audio duration.
        if respaced:
            last = respaced[-1]
            respaced[-1] = (last[0], float(total), last[2], last[3], last[4])

        planned = respaced
        planned_motion = motions

        # Pattern interrupt boost: if we have an explicit BUT beat with enough duration,
        # split it into a micro-beat + remainder so the viewer sees an abrupt change.
        boosted: list[tuple[float, float, str, str, str | None]] = []
        boosted_motion: list[str | None] = []
        for (st, en, q, h, sh), m in zip(planned, planned_motion or []):
            dur = float(en - st)
            txt = (h or "").strip().lower()
            if ("but" in txt) and dur >= 1.0 and m == "snap":
                cut = min(0.45, max(0.30, 0.33 * dur))
                mid = float(st) + float(cut)
                boosted.append((float(st), float(mid), q, "BUT…", None))
                boosted_motion.append("snap")
                boosted.append((float(mid), float(en), q, (h or "").strip(), sh))
                boosted_motion.append("zoom_in")
            else:
                boosted.append((float(st), float(en), q, (h or "").strip(), sh))
                boosted_motion.append(m)
        if boosted and boosted_motion:
            planned = boosted
            planned_motion = boosted_motion

    # Shorts Review: ensure a real hook frame exists and timings are VIDEO-time.
    if vt == "shorts_review" and timeline_duration > 0:

        has_hook_already = False
        try:
            if planned and hook_s > 0:
                has_hook_already = (abs(float(planned[0][0]) - 0.0) <= 1e-6) and (float(planned[0][1]) <= (float(hook_s) + 1e-3))
        except Exception:
            has_hook_already = False

        def _default_hook_caption() -> str:
            first_txt = ""
            try:
                first_txt = " ".join([str(s.text or "") for s in segments if float(s.start) < 3.0])
            except Exception:
                first_txt = ""
            low = first_txt.lower()
            if "oscar" in low:
                return "OSCAR BAIT?"
            if "proves" in low:
                return "THIS PROVES IT"
            return "HOT TAKE"

        if not has_hook_already:
            # Shift all planned windows forward so narration starts at hook_s.
            if planned and hook_s > 0:
                shifted: list[tuple[float, float, str, str, str | None]] = []
                for st, en, q, h, sh in planned:
                    shifted.append((float(st) + float(hook_s), float(en) + float(hook_s), q, h, sh))
                planned = shifted

            # Prepend the hook frame (force contrast by using poster query).
            hook_query = f"{effective_topic} official poster".strip() if effective_topic else "official poster"
            planned = [(0.0, float(hook_s), hook_query, _default_hook_caption(), None)] + planned
            if planned_motion is None:
                planned_motion = []
            planned_motion = ["hold"] + list(planned_motion)

            # Ensure ending frame exists and is static.
            if ending_s > 0 and timeline_duration > ending_s:
                end_start = max(float(hook_s), float(timeline_duration) - float(ending_s))
                planned.append(
                    (
                        float(end_start),
                        float(timeline_duration),
                        f"{effective_topic} official poster".strip() if effective_topic else "official poster",
                        "AGREE? 👇",
                        None,
                    )
                )
                planned_motion.append("hold")

        # Clamp last to timeline duration.
        if planned:
            # Normalize ordering + contiguity (avoid overlaps/gaps after adding ending frame).
            planned.sort(key=lambda p: float(p[0]))
            normed: list[tuple[float, float, str, str, str | None]] = []
            cur = 0.0
            for st, en, q, h, sh in planned:
                st2 = max(float(cur), float(st))
                en2 = max(st2 + 0.1, float(en))
                if st2 >= float(timeline_duration):
                    break
                en2 = min(float(timeline_duration), en2)
                normed.append((float(st2), float(en2), q, h, sh))
                cur = float(en2)
            planned = normed

            # Keep planned_motion aligned after normalization by truncation/padding.
            if planned_motion is not None:
                planned_motion = list(planned_motion)[: len(planned)]
                if len(planned_motion) < len(planned):
                    planned_motion += ["hold"] * (len(planned) - len(planned_motion))

            # Ensure exact end.
            if planned:
                last = planned[-1]
                planned[-1] = (last[0], float(timeline_duration), last[2], last[3], last[4])

    slides: list[Slide] = []
    attribution_lines: list[str] = []

    def _find_reuse_source_dir(*, audio_stem: str) -> Path | None:
        if not reuse_images:
            return None

        stem = (audio_stem or "").strip().lower()
        if not stem:
            return None

        candidates: list[tuple[float, Path]] = []
        cwd = Path.cwd()
        for d in cwd.iterdir():
            if not d.is_dir():
                continue
            name = d.name.lower()
            if not name.startswith("output"):
                continue
            if d.resolve() == out_dir.resolve():
                continue
            if not (d / "assets").is_dir():
                continue
            if not (d / "timeline.json").exists():
                continue

            # Prefer exact match using run_meta.json when present.
            meta_ok = False
            meta_path = d / "run_meta.json"
            if meta_path.exists():
                try:
                    meta = read_json(meta_path)
                    if isinstance(meta, dict):
                        if (meta.get("audio_name") == audio_path.name) or (str(meta.get("audio_stem") or "").strip().lower() == stem):
                            # If size/mtime were recorded, check them too.
                            sz = meta.get("audio_size")
                            mt = meta.get("audio_mtime")
                            if sz is None or int(sz) == int(audio_stat.st_size):
                                if mt is None or float(mt) == float(audio_stat.st_mtime):
                                    meta_ok = True
                except Exception:
                    meta_ok = False

                # If run_meta.json exists but doesn't match, do NOT fall back to name matching.
                # This prevents reusing images when the user overwrote the audio file but kept the same stem.
                if not meta_ok:
                    continue

            # Back-compat: older outputs without run_meta.json fall back to folder-name matching.
            if not meta_ok:
                if stem not in name:
                    continue
            try:
                ts = (d / "timeline.json").stat().st_mtime
            except Exception:
                ts = 0.0
            candidates.append((ts, d))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    reuse_source_dir = _find_reuse_source_dir(audio_stem=audio_path.stem)
    reuse_image_paths: list[Path] = []

    if reuse_source_dir is not None:
        src_assets = reuse_source_dir / "assets"
        # Prefer slide assets (sXX_*.png) and seed assets; avoid logos/slates.
        preferred = []
        fallback = []
        for p in sorted(src_assets.glob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            nm = p.name.lower()
            if nm.startswith("s") or nm.startswith("seed"):
                preferred.append(p)
            else:
                fallback.append(p)

        pick = preferred if preferred else fallback
        for p in pick:
            dst = assets_dir / p.name
            if not dst.exists():
                try:
                    shutil.copy2(p, dst)
                except Exception:
                    continue
            reuse_image_paths.append(dst)

        # If we successfully reused images, carry forward attribution when available.
        if reuse_image_paths and (reuse_source_dir / "attribution.txt").exists():
            try:
                shutil.copy2(reuse_source_dir / "attribution.txt", out_dir / "attribution.txt")
            except Exception:
                pass

    last_image_path: str | None = None
    last_info: dict | None = None
    last_headline_key: str | None = None
    used_source_pages: list[str] = []
    used_image_hashes: set[str] = set()
    used_source_domains: set[str] = set()

    def _domain(url: str | None) -> str | None:
        if not url:
            return None
        try:
            from urllib.parse import urlparse

            netloc = (urlparse(str(url)).netloc or "").strip().lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc or None
        except Exception:
            return None

    # Shorts Review: burn text only for the 2–3 emphasis beats (hook/pivot/cta).
    burn_in_text_cards = bool(vt == "shorts_review")

    # ── Portrait image pre-processing: blur-behind composite ──
    _is_portrait_output = int(video_height) > int(video_width) * 1.1

    def _preprocess_portrait_image(src_path: str, *, prefix: str) -> str:
        """For portrait output with landscape source images, create a blur-behind
        composite so that the full image is visible (no face/body cutoff).
        Returns the path to the preprocessed image, or *src_path* unchanged."""
        if not _is_portrait_output:
            return src_path
        try:
            from PIL import Image, ImageEnhance, ImageFilter

            img = Image.open(src_path).convert("RGB")
            src_w, src_h = img.size
            if src_w <= 0 or src_h <= 0:
                return src_path
            src_ratio = src_w / src_h
            tgt_ratio = int(video_width) / max(1, int(video_height))
            # Only composite when mismatch is extreme (landscape src → portrait out).
            if src_ratio / max(0.01, tgt_ratio) < 1.4:
                return src_path

            tw, th = int(video_width), int(video_height)

            # Background: scale to fill target, blur + darken.
            bg_scale = max(tw / src_w, th / src_h)
            bg_w, bg_h = int(src_w * bg_scale + 0.5), int(src_h * bg_scale + 0.5)
            bg = img.resize((bg_w, bg_h), Image.LANCZOS)
            bx, by = (bg_w - tw) // 2, (bg_h - th) // 2
            bg = bg.crop((bx, by, bx + tw, by + th))
            bg = bg.filter(ImageFilter.GaussianBlur(radius=30))
            bg = ImageEnhance.Brightness(bg).enhance(0.35)

            # Foreground: scale to fit (fully visible, no cropping).
            fg_scale = min(tw / src_w, th / src_h)
            fg_w, fg_h = int(src_w * fg_scale + 0.5), int(src_h * fg_scale + 0.5)
            fg = img.resize((fg_w, fg_h), Image.LANCZOS)
            fx, fy = (tw - fg_w) // 2, (th - fg_h) // 2

            canvas = bg.copy()
            canvas.paste(fg, (fx, fy))

            out_path = str(assets_dir / f"{prefix}_portrait.jpg")
            canvas.save(out_path, quality=95)
            return out_path
        except Exception:
            return src_path

    def _hash_file(path_str: str) -> str:
        b = Path(path_str).read_bytes()
        return hashlib.sha256(b).hexdigest()

    def _search_candidates(qstr: str) -> list[dict]:
        """Search images — Playwright primary, SerpAPI/Wikimedia fallback."""
        if not qstr:
            return []

        # Primary: Playwright-based Google Images scraping (free, no API key).
        try:
            from .playwright_images import playwright_google_image_search
            results = playwright_google_image_search(
                qstr, max_results=25, min_width=min_image_width,
            )
            if results:
                return results
        except Exception as e:
            print(f"[image_search] Playwright failed for {qstr!r}: {e}")

        # Fallback: SerpAPI if available and configured.
        if image_provider == "serpapi" and serpapi_api_key:
            from .serpapi_provider import search_commons_candidates_via_serpapi
            return search_commons_candidates_via_serpapi(
                qstr,
                api_key=serpapi_api_key,
                min_width=min_image_width,
                max_results=25,
            )
        if image_provider == "google_images" and serpapi_api_key:
            from .serpapi_provider import search_google_images_candidates_via_serpapi
            return search_google_images_candidates_via_serpapi(
                qstr,
                api_key=serpapi_api_key,
                min_width=min_image_width,
                max_results=25,
            )

        # Last resort: DuckDuckGo.
        try:
            from .playwright_images import _ddg_image_fallback
            results = _ddg_image_fallback(qstr, max_results=25, min_width=min_image_width)
            if results:
                return results
        except Exception:
            pass

        # Wikimedia Commons (always available).
        return search_commons_images(qstr, min_width=min_image_width, max_results=25)

    def _tokenize(s: str) -> set[str]:
        try:
            import re

            return {t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if len(t) >= 3}
        except Exception:
            return set()

    def _candidate_score(c: dict, *, qstr: str, target_ratio: float) -> float:
        """Heuristic score: higher is better."""
        try:
            import math

            w = float(c.get("width") or 0.0)
            h = float(c.get("height") or 0.0)
            area = max(1.0, w * h)
            score = 0.0

            # Prefer higher-resolution images.
            score += min(4.0, math.log1p(area / 1_000_000.0) * 2.0)

            # Prefer aspect ratios closer to the output video frame.
            if w > 0 and h > 0 and target_ratio > 0:
                aspect = w / h
                ratio_penalty = abs(math.log(max(1e-6, aspect / target_ratio)))
                # For portrait output, heavily penalize landscape images (extreme crop).
                if _is_portrait_output and aspect > 1.2:
                    ratio_penalty *= 2.5
                score -= ratio_penalty

            title = str(c.get("title") or "")
            page = str(c.get("page_url") or "")
            blob = (title + " " + page).lower()

            # Penalize common low-value results.
            bad = ["poster", "logo", "wordmark", "cover", "album", "soundtrack", "dvd", "blu-ray", "bluray"]
            if any(b in blob for b in bad):
                score -= 2.5

            # Penalize production / press / BTS results — not actual movie frames.
            bad_prod = ["behind the scenes", "on set", "press", "premiere", "red carpet",
                        "photocall", "production still", "interview", "arrivals", "junket"]
            if any(b in blob for b in bad_prod):
                score -= 2.0

            # Bonus for scene still / portrait hints.
            good = ["still", "scene", "screencap", "frame", "screenshot", "close", "face"]
            if any(g in blob for g in good):
                score += 0.8

            # Query token overlap.
            qt = _tokenize(qstr)
            if qt:
                tt = _tokenize(title)
                overlap = len(qt & tt)
                score += min(1.6, overlap * 0.25)

            # Light domain penalties for spammy sources (only affects google_images provider typically).
            dom = _domain(page) or ""
            if any(d in dom for d in ("pinterest.", "pinimg.", "shutterstock.", "alamy.", "gettyimages.") ):
                score -= 1.2

            return float(score)
        except Exception:
            return 0.0

    def _rerank_candidates(candidates: list[dict], *, qstr: str) -> list[dict]:
        if not candidates:
            return []
        target_ratio = float(video_width) / float(video_height) if float(video_height) > 0 else 16.0 / 9.0
        scored: list[tuple[tuple[int, int, float], dict]] = []
        for c in candidates:
            page = str(c.get("page_url") or "")
            page_used = 1 if (page and page in used_source_pages) else 0
            dom = _domain(page)
            dom_used = 1 if (dom and dom in used_source_domains) else 0
            s = _candidate_score(c, qstr=qstr, target_ratio=target_ratio)
            scored.append(((page_used, dom_used, -float(s)), c))
        scored.sort(key=lambda x: x[0])
        return [c for _, c in scored]

    def _window_text(start: float, end: float) -> str:
        # Shorts Review uses VIDEO-time windows; map back to ORIGINAL audio-time for transcript lookup.
        if vt == "shorts_review" and hook_s > 0:
            def _video_to_audio(t: float) -> float:
                tt = float(t)
                # Undo the hook offset.
                tt = tt - float(hook_s)
                # Undo any pivot pauses (piecewise: during the pause, audio time is frozen).
                for ps, pd in pause_schedule:
                    if float(t) <= float(ps):
                        break
                    spent = min(float(pd), max(0.0, (float(t) - float(ps))))
                    tt -= float(spent)
                return max(0.0, float(tt))

            start = _video_to_audio(float(start))
            end = _video_to_audio(float(end))
        text = ""
        for s in merged:
            if s.end <= start:
                continue
            if s.start >= end:
                break
            text += " " + s.text
        return text.strip()

    def _build_caption_spans(segments_in: list, *, min_hold: float = 0.8) -> list[dict]:
        """Build immutable caption spans from transcript only.

        Short segments are merged into the previous span for readability.
        """

        spans: list[dict] = []
        for s in segments_in or []:
            try:
                st = float(getattr(s, "start"))
                en = float(getattr(s, "end"))
                txt = str(getattr(s, "text") or "").strip()
            except Exception:
                continue
            if not txt or en <= st:
                continue

            dur = float(en - st)
            if dur < float(min_hold) and spans:
                spans[-1]["end"] = max(float(spans[-1]["end"]), float(en))
                spans[-1]["text"] = (str(spans[-1]["text"]).rstrip() + " " + txt).strip()
                continue

            spans.append({"start": float(st), "end": float(en), "text": txt})

        spans.sort(key=lambda x: float(x["start"]))
        return spans

    def _audio_to_video_time(t_audio: float, *, hook_seconds: float, insertions: list[tuple[float, float]]) -> float:
        """Map ORIGINAL audio time -> VIDEO time (hook offset + inserted pivot silences)."""

        tt = max(0.0, float(t_audio)) + max(0.0, float(hook_seconds))
        for at, d in (insertions or []):
            try:
                if float(t_audio) >= float(at):
                    tt += max(0.0, float(d))
            except Exception:
                continue
        return float(tt)

    # Shorts Review: Visual Rules (strict).
    # - Keep ONLY hook title + loop CTA text (no other overlays)
    # - 30–40s => 6–8 images max (one image = one idea)
    # - Holds: min 2.0s; hook >= 3.0s; sweet spot 2.5–3.5s when possible
    # - Transitions: ONLY one motion style per Short (default: static hold)
    if vt == "shorts_review" and planned and timeline_duration > 0:
        if planned_motion is None:
            planned_motion = ["hold"] * len(planned)

        # Ensure we do NOT insert pivot pauses/cards in this mode.
        audio_pause_insertions = []
        pause_schedule = []

        # IMPORTANT: some upstream shorts_review planners may extend `timeline_duration` to account
        # for inserted pivot silences. In strict visual mode we disallow pivot pauses, so we must
        # ensure the timeline ends exactly when the audio ends (plus hook offset).
        timeline_duration = float(audio_duration) + float(hook_s)

        def _clean_text(t: str) -> str:
            return " ".join((t or "").split()).strip()

        def _sanitize_hook_text(t: str) -> str:
            # Per spec: hook overlay is fixed for review shorts.
            return "Honest Review"

        def _sanitize_cta_text(_t: str) -> str:
            # Per spec: outro overlay is fixed.
            return "Agree?"

        # Identify hook + CTA windows.
        hook_idx: int | None = None
        cta_idx: int | None = None
        for i, (st, en, _q, _h, _sh) in enumerate(planned):
            if hook_idx is None and float(st) <= 1e-6 and float(en) <= (float(hook_s) + 1e-3):
                hook_idx = int(i)
            if float(en) >= (float(timeline_duration) - 1e-3) and ((float(timeline_duration) - float(st)) <= (float(ending_s) + 0.25)):
                cta_idx = int(i)

        # Apply rule: only hook/cta may have text; all others empty.
        new_planned2: list[tuple[float, float, str, str, str | None]] = []
        for i, (st, en, q, h, _sh) in enumerate(planned):
            if hook_idx is not None and int(i) == int(hook_idx):
                new_planned2.append((float(st), float(en), q, _sanitize_hook_text(h), None))
                planned_motion[i] = "hold"
            elif cta_idx is not None and int(i) == int(cta_idx):
                new_planned2.append((float(st), float(en), q, _sanitize_cta_text(h), None))
                planned_motion[i] = "hold"
            else:
                new_planned2.append((float(st), float(en), q, "", None))
        planned = new_planned2

        # Hard constraint: max 2 text windows (hook + CTA).
        text_idxs = [i for i, (_st, _en, _q, h, _sh) in enumerate(planned) if _clean_text(h)]
        if len(text_idxs) > 2:
            keep: set[int] = set()
            if hook_idx is not None:
                keep.add(int(hook_idx))
            if cta_idx is not None:
                keep.add(int(cta_idx))
            for i in text_idxs:
                if int(i) not in keep:
                    st, en, q, _h, _sh = planned[i]
                    planned[i] = (float(st), float(en), q, "", None)

        # Visual cadence rule: distribute images throughout the short.
        # Non-text windows become a small number of idea windows.
        MIN_HOLD_S = 2.0
        HOOK_MIN_S = 3.0
        MOTION_STYLE = "hold"  # single motion style for the entire short

        planned_src = list(planned)

        def _query_at(t: float) -> str:
            """Pick query from the original plan covering time t."""

            tt = float(t)
            for (st, en, q, _h, _sh) in planned_src:
                if float(st) <= tt <= float(en):
                    return str(q or "")
            # Fallback: nearest previous.
            best_q = ""
            for (st, en, q, _h, _sh) in planned_src:
                if float(en) <= tt:
                    best_q = str(q or "")
                else:
                    break
            return best_q

        # Find special windows (hook/cta) after enforcement.
        hook_win: tuple[float, float, str, str, str | None] | None = None
        for st, en, q, h, sh in planned:
            h_clean = _clean_text(h)
            if hook_win is None and float(st) <= 1e-6 and float(en) <= (float(hook_s) + 1e-3):
                hook_win = (float(st), float(en), str(q or ""), str(h_clean), sh)
                continue

        def _append_window(st: float, en: float, q: str, h: str, m: str | None) -> None:
            planned_new.append((float(st), float(en), str(q or ""), str(h or ""), None))
            motion_new.append(MOTION_STYLE)

        planned_new: list[tuple[float, float, str, str, str | None]] = []
        motion_new: list[str | None] = []

        # Always start with hook (if present).
        cur_t = 0.0
        if hook_win is not None:
            # Enforce hook minimum hold.
            hk_st = float(hook_win[0])
            hk_en = float(hook_win[1])
            if (hk_en - hk_st) < float(HOOK_MIN_S):
                hk_en = float(hk_st) + float(HOOK_MIN_S)
            hk_en = min(float(timeline_duration), hk_en)
            _append_window(hk_st, hk_en, hook_win[2], hook_win[3], MOTION_STYLE)
            cur_t = float(hk_en)
        else:
            cur_t = 0.0

        # Speech portion ends before CTA tail.
        speech_end = max(float(cur_t), float(timeline_duration) - float(ending_s))

        def _fill_idea_windows(a: float, b: float, *, images: int) -> None:
            a2 = float(a)
            b2 = float(b)
            if images <= 0 or b2 <= a2 + 1e-6:
                return

            # Ensure minimum hold; if the interval is too short, reduce window count.
            span = float(b2 - a2)
            max_images_by_min_hold = max(1, int(span // float(MIN_HOLD_S)))
            n = int(min(int(images), int(max_images_by_min_hold)))
            n = max(1, n)

            dur = float(span) / float(n)
            # Hard minimum hold.
            if dur < float(MIN_HOLD_S):
                dur = float(MIN_HOLD_S)

            t = float(a2)
            for i in range(int(n)):
                nxt = float(b2) if (i == int(n) - 1) else min(float(b2), float(t + dur))
                mid = float(t + (nxt - t) / 2.0)
                q = _query_at(mid)
                _append_window(float(t), float(nxt), q, "", MOTION_STYLE)
                t = float(nxt)

        # Determine target image count.
        # Total images includes hook + ending CTA.
        total_target = _shorts_review_target_total_images(float(audio_duration))
        specials = 0
        if hook_win is not None and float(hook_win[1]) > float(hook_win[0]):
            specials += 1
        # CTA always exists (we build it deterministically below).
        if float(ending_s) > 0.0 and float(timeline_duration) > float(ending_s):
            specials += 1
        mid_images = max(0, int(total_target) - int(specials))

        # Fill between hook end and CTA start with a small number of idea windows.
        _fill_idea_windows(float(cur_t), float(speech_end), images=int(mid_images))

        # Deterministic CTA window at the end.
        if float(ending_s) > 0.0 and float(timeline_duration) > float(ending_s):
            cta_start = max(float(speech_end), float(timeline_duration) - float(ending_s))
            # Choose a query near the end for the CTA background.
            q_cta = _query_at(float(cta_start + (float(timeline_duration) - float(cta_start)) / 2.0))
            _append_window(float(cta_start), float(timeline_duration), str(q_cta or ""), "Agree?", MOTION_STYLE)
        else:
            # Tiny audio: just ensure we end at timeline_duration.
            _fill_idea_windows(float(speech_end), float(timeline_duration), images=1)

        # Replace planned with the uniform-cadence version.
        planned = planned_new
        planned_motion = motion_new

        # Final enforcement: one motion style for the entire short.
        if planned_motion is not None:
            planned_motion = [MOTION_STYLE] * len(planned_motion)

    # Shorts Review: speech-driven caption snapping (legacy).
    # Disabled: we only use emphasis text beats (hook/pivot/cta), not sentence captions.
    if False and vt == "shorts_review" and planned and planned_motion and merged:
        caption_spans = _build_caption_spans(list(merged), min_hold=0.8)

        ins = [(max(0.0, float(t)), max(0.0, float(d))) for (t, d) in (audio_pause_insertions or []) if float(d) > 1e-6]
        ins.sort(key=lambda x: x[0])

        caption_spans_v: list[dict] = []
        for c in caption_spans:
            st_a = float(c["start"])
            en_a = float(c["end"])
            st_v = _audio_to_video_time(st_a, hook_seconds=float(hook_s), insertions=ins)
            en_v = _audio_to_video_time(en_a, hook_seconds=float(hook_s), insertions=ins)
            if en_v <= st_v:
                continue
            txt = str(c["text"] or "").strip()
            if not txt:
                continue
            caption_spans_v.append({"start": float(st_v), "end": float(en_v), "text": txt})

        # Keep special non-speech cards (hook, pivot silence card, ending CTA).
        specials: list[tuple[float, float, str, str, str | None]] = []
        specials_motion: list[str | None] = []
        for (st, en, q, h, sh), m in zip(planned, planned_motion):
            h_clean = " ".join((h or "").split()).strip()
            # Hook window.
            if (float(st) <= 1e-6) and (float(en) <= (float(hook_s) + 1e-3)):
                specials.append((float(st), float(en), str(q or ""), str(h or "").strip(), sh))
                specials_motion.append("hold")
                continue
            # Pivot silence card.
            if h_clean == "BUT…":
                specials.append((float(st), float(en), str(q or ""), "BUT…", None))
                specials_motion.append("snap")
                continue
            # Ending loop CTA.
            if h_clean == "AGREE? 👇":
                specials.append((float(st), float(en), str(q or ""), "AGREE? 👇", None))
                specials_motion.append("hold")
                continue

        # Choose a visual query/motion for each caption span from the planned window covering its midpoint.
        planned_sorted = list(zip(planned, planned_motion))
        planned_sorted.sort(key=lambda x: float(x[0][0]))

        speech_windows: list[tuple[float, float, str, str, str | None]] = []
        speech_motion: list[str | None] = []

        wi = 0
        for c in caption_spans_v:
            st = float(c["start"])
            en = float(c["end"])
            if en <= st:
                continue
            mid = float(st + (en - st) / 2.0)

            while wi + 1 < len(planned_sorted) and float(planned_sorted[wi][0][1]) <= mid:
                wi += 1

            chosen_q = ""
            chosen_m: str | None = "hold"
            for off in (0, -1, 1, -2, 2):
                j = wi + off
                if j < 0 or j >= len(planned_sorted):
                    continue
                (wst, wen, wq, _wh, _wsh), wm = planned_sorted[j]
                if float(wst) <= mid <= float(wen):
                    chosen_q = str(wq or "")
                    chosen_m = wm
                    break
                if chosen_q == "":
                    chosen_q = str(wq or "")
                    chosen_m = wm

            cap_txt = str(c["text"] or "").strip()
            if not cap_txt:
                continue

            # Hard minimum caption duration: merge micro-spans into previous.
            if speech_windows and (float(en - st) < 0.7):
                prev = speech_windows[-1]
                speech_windows[-1] = (
                    float(prev[0]),
                    float(en),
                    prev[2],
                    (str(prev[3]).rstrip() + " " + cap_txt).strip(),
                    prev[4],
                )
                continue

            speech_windows.append((float(st), float(en), str(chosen_q or ""), cap_txt, None))
            speech_motion.append(chosen_m)

        combined = list(zip(specials, specials_motion)) + list(zip(speech_windows, speech_motion))
        combined.sort(key=lambda x: float(x[0][0]))

        planned = [w for w, _m in combined]
        planned_motion = [(_m if _m is not None else "hold") for _w, _m in combined]

    # Shorts Review motion budget:
    # - cap overall motion frequency
    # - add snap cooldown (no motion the very next beat)
    # - ignore snap motion on ultra-short windows
    if vt == "shorts_review" and planned and planned_motion:
        MAX_MOTION_RATIO = 0.4
        motion_count = 0
        snap_cooldown = 0
        n = max(1, len(planned_motion))

        # Always keep hook + ending static.
        planned_motion[0] = "hold"
        planned_motion[-1] = "hold"

        for mi, m in enumerate(list(planned_motion)):
            if mi <= 0 or mi >= (len(planned_motion) - 1):
                continue

            # Preserve the dedicated pivot emphasis card motion.
            try:
                h_clean = " ".join((planned[mi][3] or "").split()).strip()
            except Exception:
                h_clean = ""
            if h_clean in {"BUT…", "WAIT…", "HERE'S WHY", "HERE’S WHY"}:
                planned_motion[mi] = "snap"
                snap_cooldown = 1
                continue

            if snap_cooldown > 0:
                planned_motion[mi] = "hold"
                snap_cooldown -= 1
                continue

            try:
                dur = float(planned[mi][1]) - float(planned[mi][0])
            except Exception:
                dur = 999.0

            mm = (m or "hold").strip().lower()
            if mm in {"", "none"}:
                mm = "hold"

            # Snap must not be a micro-blip.
            if mm == "snap" and dur < 0.3:
                planned_motion[mi] = "hold"
                continue

            if mm != "hold":
                motion_count += 1
                if (motion_count / float(n)) > float(MAX_MOTION_RATIO):
                    planned_motion[mi] = "hold"
                    continue

            if mm == "snap":
                snap_cooldown = 1

    def _pick_info_with_llm(*, slide_query: str, window_text: str, candidates: list[dict]) -> dict | None:
        if not candidates:
            return None
        if not (use_llm and llm_pick_images):
            # Deterministic fallback: pick first candidate not already used.
            for c in candidates:
                if c.get("page_url") and c.get("page_url") in used_source_pages:
                    continue
                dom = _domain(c.get("page_url"))
                if dom and dom in used_source_domains:
                    continue
                return c
            return candidates[0]

        from .llm_storyboard import pick_image_with_llm

        idx = pick_image_with_llm(
            slide_query=slide_query,
            window_text=window_text,
            candidates=candidates,
            used_source_pages=used_source_pages,
            model=llm_model,
        )
        return candidates[idx]

    def _download_unique(info: dict, *, prefix: str) -> str | None:
        dl_url = info.get("original_url") or info.get("image_url")
        if not dl_url:
            return None
        p = download_image(dl_url, assets_dir, prefix=prefix)
        h = _hash_file(str(p))
        if h in used_image_hashes:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
            return None
        used_image_hashes.add(h)
        return str(p)

    # Seed image: try to download at least one image for the topic.
    seed_topic = effective_topic or (topic or "").strip() or None
    _cached_seed_candidates: list[dict] = []  # reused by pool block below
    if (reuse_source_dir is None) and seed_topic:
        try:
            seed_candidates = _search_candidates(seed_topic)
            seed_candidates = _rerank_candidates(seed_candidates, qstr=seed_topic)
            _cached_seed_candidates = seed_candidates
            seed = seed_candidates[0] if seed_candidates else None
            if seed and seed.get("image_url"):
                p = download_image(seed["image_url"], assets_dir, prefix="seed")
                last_image_path = str(p)
                last_info = seed
                if seed.get("page_url"):
                    used_source_pages.append(str(seed.get("page_url")))
                try:
                    used_image_hashes.add(_hash_file(last_image_path))
                except Exception:
                    pass
        except Exception:
            last_image_path = None
            last_info = None

    hook_raw_bg_image: str | None = None
    hook_info_for_reuse: dict | None = None

    # ── Review still-pool: single query, top 20 images ──
    # Skipped when the caller already provided pool_image_paths (e.g. UI image picker).
    if pool_image_paths:
        _provided = [Path(p) for p in pool_image_paths if Path(p).exists()]
        if _provided:
            import random as _rng_review
            _rng_review.shuffle(_provided)
            reuse_image_paths = _provided
            print(f"[review] Using {len(reuse_image_paths)} UI-selected images")
    elif vt == "review" and not reuse_image_paths and seed_topic:
        import random as _rng_review

        _still_query = seed_topic.strip()
        _still_pool_paths: list[Path] = []
        # Reuse results already fetched for the seed image (same query) to avoid a
        # duplicate search.
        if _cached_seed_candidates:
            _pool_cands = _cached_seed_candidates
        else:
            try:
                _pool_cands = _search_candidates(_still_query)
            except Exception:
                _pool_cands = []
            _pool_cands = _rerank_candidates(_pool_cands, qstr=_still_query)
        for _cand in _pool_cands[:20]:
            try:
                _p = _download_unique(_cand, prefix=f"pool{len(_still_pool_paths):02d}")
            except Exception:
                _p = None
            if _p:
                _still_pool_paths.append(Path(_p))
                if _cand.get("page_url"):
                    used_source_pages.append(str(_cand.get("page_url")))
                    _d = _domain(str(_cand.get("page_url")))
                    if _d:
                        used_source_domains.add(_d)
        if _still_pool_paths:
            _rng_review.shuffle(_still_pool_paths)
            reuse_image_paths = _still_pool_paths
            print(f"[review] Pre-fetched {len(_still_pool_paths)} stills for {_still_query!r}")

    for i, (start, end, llm_query, headline, subhead) in enumerate(tqdm(planned, desc="Finding images")):
        # Caption integrity (Shorts Review): if the on-screen caption doesn't change,
        # the background image must not change either (motion is allowed; image swaps are not).
        headline_key = " ".join((headline or "").split()).strip()

        # Emphasis beat kinds (only these may render visual text).
        is_hook = (float(start) <= 1e-6) and (float(end) <= (float(hook_s) + 1e-3))
        is_cta = (float(end) >= (float(timeline_duration) - 1e-3)) and ((float(timeline_duration) - float(start)) <= (float(ending_s) + 0.25))
        is_pivot = headline_key in {"BUT…", "WAIT…", "HERE'S WHY", "HERE’S WHY"}

        if vt == "shorts_review" and headline_key and last_headline_key and (headline_key == last_headline_key):
            if last_image_path is not None:
                # Same caption => same image => no motion.
                slide_motion = "hold"

                raw_image_path = str(last_image_path)
                slide_image_path = raw_image_path

                if burn_in_text_cards and headline_key and (is_hook or is_pivot or is_cta):
                    try:
                        from .slide_cards import SlideCardSpec, render_slide_card

                        layout = "shorts_emphasis_hook" if is_hook else ("shorts_emphasis_pivot" if is_pivot else "shorts_emphasis_cta")
                        card_path = assets_dir / f"card_{i:02d}.png"
                        render_slide_card(
                            background_image=str(raw_image_path),
                            out_path=str(card_path),
                            spec=SlideCardSpec(headline=str(headline_key), subhead=None),
                            width=int(video_width),
                            height=int(video_height),
                            layout=layout,
                        )
                        slide_image_path = str(card_path)
                    except Exception:
                        slide_image_path = raw_image_path

                slide = Slide(
                    start=float(start),
                    end=float(end),
                    image_path=slide_image_path,
                    query=(str(llm_query or "").strip() or (last_info or {}).get("_query") or ""),
                    headline=(headline_key or None),
                    subhead=None,
                    source_page=(last_info or {}).get("page_url"),
                    image_url=(last_info or {}).get("image_url"),
                    license_name=(last_info or {}).get("license_name"),
                    license_url=(last_info or {}).get("license_url"),
                    attribution=(last_info or {}).get("attribution"),
                    motion=slide_motion,
                    window_text=None,
                    window_keywords=None,
                    queries_tried=None,
                )

                slides.append(slide)
                last_headline_key = headline_key
                if slide.source_page or slide.license_name:
                    attribution_lines.append(
                        f"{Path(raw_image_path).name} | {slide.license_name or ''} | {slide.license_url or ''} | {slide.source_page or ''} | {slide.attribution or ''}".strip()
                    )
                continue

        # Build a query that is:
        # 1) anchored on the explicit topic/visual subject (movie/show), and
        # 2) driven by transcript-window keywords so the still matches what's being said.
        q_llm_raw = (llm_query or "").strip()
        wtext = _window_text(float(start), float(end))
        # Widen the keyword window slightly to reduce empty/low-signal keyword sets.
        kw_text = _window_text(max(0.0, float(start) - 1.0), float(end) + 1.0)
        debug_window_text = (kw_text or wtext or "").strip()
        if len(debug_window_text) > 800:
            debug_window_text = debug_window_text[:800].rstrip() + "…"
        keywords = extract_keywords(debug_window_text, max_words=8)

        # Prefer the explicitly provided visual subject/topic anchor when available.
        anchor = (effective_topic or topic or "").strip() or None

        # Primary query should be transcript-context within the anchored subject.
        q_context = " ".join(([anchor] if anchor else []) + (keywords[:5] if keywords else [])).strip() if (anchor or keywords) else ""

        # Let the LLM propose a few concrete visual search phrases (e.g. "<movie> guitar scene").
        llm_suggested_queries: list[str] = []
        if use_llm and anchor and debug_window_text:
            try:
                from .llm_storyboard import suggest_image_search_queries_with_llm

                llm_suggested_queries = suggest_image_search_queries_with_llm(
                    anchor=str(anchor),
                    window_text=str(debug_window_text),
                    topic_type=topic_type,
                    video_type=vt,
                    model=llm_model,
                    max_queries=5,
                )
            except Exception:
                llm_suggested_queries = []

        # Secondary query: LLM-suggested hints, but still anchored.
        q_llm = q_llm_raw
        _anchor_llm = vt in {"explainer", "shorts", "shorts_review"}
        if _anchor_llm and anchor:
            if q_llm:
                if anchor.lower() not in q_llm.lower():
                    q_llm = f"{anchor} {q_llm}".strip()
            else:
                q_llm = ""

        # If this looks like TV content, bias toward episode stills/cast.
        _tv_enrich = vt in {"explainer", "shorts", "shorts_review"}
        if _tv_enrich and (topic_type == "tv_show"):
            if q_context and not any(k in q_context.lower() for k in ("still", "stills", "cast", "scene", "episode")):
                q_context = f"{q_context} TV series scene still".strip()
            if q_llm and not any(k in q_llm.lower() for k in ("still", "stills", "cast", "scene", "episode")):
                q_llm = f"{q_llm} TV series scene still".strip()

        # Shorts Review or portrait-review: bias toward in-scene imagery.
        _enrich_queries = vt in {"shorts_review"}
        if _enrich_queries:
            # Rotate visual suffixes per slide to encourage image variety.
            _VISUAL_SUFFIXES = [
                "scene still", "character close up", "cinematic frame",
                "dramatic moment", "promotional still", "behind the scenes",
                "cast photo", "key scene", "portrait shot", "wide shot",
            ]
            _suffix = _VISUAL_SUFFIXES[i % len(_VISUAL_SUFFIXES)]
            if q_context and not any(k in q_context.lower() for k in ("still", "stills", "scene", "screencap", "frame", "close", "portrait", "wide", "cinematic")):
                q_context = f"{q_context} {_suffix}".strip()

            # Avoid posters for non-emphasis beats; use scene stills instead.
            if not (is_hook or is_cta):
                if q_llm and any(k in q_llm.lower() for k in ("poster", "official poster")):
                    q_llm = q_llm.replace("official poster", "").replace("Official poster", "")
                    q_llm = q_llm.replace("poster", "").replace("Poster", "")
                    q_llm = " ".join(q_llm.split()).strip()
                # If LLM query becomes empty/weak, just drop it.
                if q_llm and len(q_llm.split()) <= (1 if anchor else 0):
                    q_llm = ""

        if storyboard == "llm" and not (q_context or q_llm):
            raise RuntimeError("LLM slide query is empty")

        q_main = q_context or q_llm or (anchor or "")

        # Avoid overly-specific queries causing 0 results, but always include a transcript-context attempt.
        queries: list[str] = []
        # Try LLM-suggested transcript-context queries first.
        for qq in llm_suggested_queries:
            if qq:
                queries.append(qq)
        # Then try our deterministic transcript-context query.
        if q_context:
            queries.append(q_context)
        # Then try the storyboard/LLM-provided hint (still anchored), if any.
        if q_llm and q_llm not in queries:
            queries.append(q_llm)
        # Then broad fallbacks.
        if anchor and anchor not in queries:
            queries.append(anchor)
        if topic and topic not in queries:
            queries.append(topic)
        if keywords:
            kw_only = " ".join(keywords[:5]).strip()
            if kw_only and kw_only not in queries:
                queries.append(kw_only)

        # de-dup while preserving order
        seen_q: set[str] = set()
        queries = [x for x in queries if x and not (x in seen_q or seen_q.add(x))]

        info = None
        image_path = None
        winning_query = q_main

        # Reuse existing downloaded assets when possible (no SerpAPI calls).
        if reuse_image_paths:
            p = reuse_image_paths[i % len(reuse_image_paths)]
            image_path = str(p)

        for q in ([] if image_path else queries):
            try:
                candidates = _search_candidates(q)
            except Exception:
                candidates = []

            candidates = _rerank_candidates(candidates, qstr=q)

            picked = None
            try:
                picked = _pick_info_with_llm(slide_query=q, window_text=wtext, candidates=candidates)
            except Exception:
                picked = candidates[0] if candidates else None

            if picked:
                picked["_query"] = q
                winning_query = q
                # Try to download a unique image; if duplicate, try the next candidates.
                attempts = [picked] + [c for c in candidates if c is not picked]
                for cand in attempts[:10]:
                    try:
                        path_str = _download_unique(cand, prefix=f"s{i:02d}")
                    except Exception:
                        path_str = None
                    if path_str:
                        info = cand
                        image_path = path_str
                        if cand.get("page_url"):
                            used_source_pages.append(str(cand.get("page_url")))
                            dom = _domain(str(cand.get("page_url")))
                            if dom:
                                used_source_domains.add(dom)
                        last_info = cand
                        break

            if image_path:
                break

        if not image_path:
            # No image found; reuse last successful image (keeps timing valid)
            if last_image_path is None:
                continue
            image_path = last_image_path
            # Carry forward metadata so timeline/attribution stays accurate.
            info = last_info

        raw_image_path = str(image_path)
        # Portrait pre-processing: create blur-behind composite to avoid face/body cutoff.
        raw_image_path = _preprocess_portrait_image(raw_image_path, prefix=f"s{i:02d}")
        last_image_path = raw_image_path
        if headline_key:
            last_headline_key = headline_key

        slide_motion = (planned_motion[i] if planned_motion and i < len(planned_motion) else None)

        # Persist hook background for CTA reuse (palette match).
        if is_hook and hook_raw_bg_image is None:
            hook_raw_bg_image = raw_image_path
            hook_info_for_reuse = info

        slide_image_path = raw_image_path
        # Burn emphasis text into a card image (Shorts Review only).
        if burn_in_text_cards and headline_key and (is_hook or is_pivot or is_cta):
            try:
                from .slide_cards import SlideCardSpec, render_slide_card

                layout = "shorts_emphasis_hook" if is_hook else ("shorts_emphasis_pivot" if is_pivot else "shorts_emphasis_cta")
                card_path = assets_dir / f"card_{i:02d}.png"
                render_slide_card(
                    background_image=str(raw_image_path),
                    out_path=str(card_path),
                    spec=SlideCardSpec(headline=str(headline_key), subhead=None),
                    width=int(video_width),
                    height=int(video_height),
                    layout=layout,
                )
                slide_image_path = str(card_path)
            except Exception:
                slide_image_path = raw_image_path

        slide = Slide(
            start=float(start),
            end=float(end),
            image_path=slide_image_path,
            query=str(winning_query or q_main),
            headline=(headline_key or None),
            subhead=None,
            source_page=(info or {}).get("page_url"),
            image_url=(info or {}).get("image_url"),
            license_name=(info or {}).get("license_name"),
            license_url=(info or {}).get("license_url"),
            attribution=(info or {}).get("attribution"),
            motion=slide_motion,
            window_text=(debug_window_text or None),
            window_keywords=(keywords or None),
            queries_tried=(queries or None),
        )
        slides.append(slide)

        if slide.source_page or slide.license_name:
            attribution_lines.append(
                f"{Path(raw_image_path).name} | {slide.license_name or ''} | {slide.license_url or ''} | {slide.source_page or ''} | {slide.attribution or ''}".strip()
            )

    if not slides:
        raise RuntimeError(
            "No images could be found/downloaded for the selected segments. Try a different --topic, increase --max-images, or lower --min-image-width."
        )

    # ── Video clip mixing: official trailers → 7s clips → fill every slide ──
    # Skip this path when --clip-queries or --clip-research is used (handled below).
    if mix_video_clips and vt in {"review", "review_long"} and not clip_queries and not clip_research:
        try:
            from .clip_tools import (
                search_video_clips,
                download_clip,
                download_clip_section,
                trim_clip,
                get_video_duration as _rv_clip_dur,
            )
            import random as _rv_rnd

            clip_dir = ensure_dir(out_dir / "clips")
            _rv_raw_dir = ensure_dir(clip_dir / "raw")
            _clip_topic = effective_topic or audio_path.stem
            _rv_clip_secs = 7.0
            _rv_skip_secs = 12.0  # skip opening credits

            _REJECT_RV = {
                "reaction", "react", "review", "explained", "breakdown", "analysis",
                "commentary", "opinion", "discuss", "theory", "theories", "ranking",
                "tier list", "video essay", "response", "rant", "hot take",
            }

            def _is_official_rv(t: str) -> bool:
                tl = t.lower()
                for rw in _REJECT_RV:
                    if rw in tl:
                        return False
                return True

            # ── Reuse raw trailers from a prior run if available ──
            _rv_raw_trailers: list[tuple[Path, float]] = []
            _rv_seen_files: set[str] = set()

            _prior_raw_dirs = []
            if reuse_source_dir is not None:
                _prior_raw_dirs.append(reuse_source_dir / "clips" / "raw")
            _prior_raw_dirs.append(_rv_raw_dir)

            for _prd in _prior_raw_dirs:
                if not _prd.is_dir():
                    continue
                for _rf in sorted(_prd.glob("trailer_*.mp4")):
                    if _rf.name in _rv_seen_files:
                        continue
                    _rd = _rv_clip_dur(_rf)
                    if _rd >= 15.0:
                        _dst = _rv_raw_dir / _rf.name
                        if not _dst.exists() and _rf.resolve() != _dst.resolve():
                            import shutil as _rv_sh
                            _rv_sh.copy2(_rf, _dst)
                        _rv_raw_trailers.append((_dst, _rd))
                        _rv_seen_files.add(_rf.name)
                        print(f"[clip_mix] Reused trailer: {_rf.name} ({_rd:.0f}s)")
                        if len(_rv_raw_trailers) >= 3:
                            break
                if len(_rv_raw_trailers) >= 3:
                    break

            # Download fresh trailers if needed.
            if len(_rv_raw_trailers) < 3:
                print(f"[clip_mix] Searching for official trailers of '{_clip_topic}'…")
                _seen_urls: set[str] = set()
                for _tq in [
                    f"{_clip_topic} official trailer",
                    f"{_clip_topic} official trailer 2",
                    f"{_clip_topic} trailer HD",
                    f"{_clip_topic} trailer",
                ]:
                    if len(_rv_raw_trailers) >= 3:
                        break
                    try:
                        _results = search_video_clips(
                            _tq, max_results=10, preferred_max_duration=600.0,
                            sort_by_views=False,
                        )
                        _results = [r for r in _results if _is_official_rv(r.title) and r.url not in _seen_urls]
                        for _ch in _results:
                            if len(_rv_raw_trailers) >= 3:
                                break
                            _tidx = len(_rv_raw_trailers) + 1
                            _rp = download_clip_section(
                                _ch.url, _rv_raw_dir,
                                start=0.0, end=300.0,
                                prefix=f"trailer_{_tidx:02d}",
                            )
                            if _rp is None:
                                _rp = download_clip(_ch.url, _rv_raw_dir, prefix=f"trailer_{_tidx:02d}", max_duration=400.0)
                            if _rp and _rv_clip_dur(_rp) >= 15.0:
                                _seen_urls.add(_ch.url)
                                _rd = _rv_clip_dur(_rp)
                                _rv_raw_trailers.append((_rp, _rd))
                                print(f"[clip_mix] Downloaded trailer {_tidx}: {_ch.title[:60]} ({_rd:.0f}s)")
                    except Exception as _te:
                        print(f"[clip_mix] trailer search error: {_te}")

            # Cut all non-overlapping 7s clips from each trailer (skip first 12s).
            _rv_all_clips: list[Path] = []
            for _tidx, (_rp, _rd) in enumerate(_rv_raw_trailers):
                _seg = _rv_skip_secs
                _end_limit = _rd - 2.0
                _cidx = 0
                while _seg + _rv_clip_secs <= _end_limit:
                    _out = clip_dir / f"rv_t{_tidx+1:02d}_c{_cidx+1:03d}.mp4"
                    if not (_out.exists() and _rv_clip_dur(_out) > 0):
                        try:
                            trim_clip(_rp, _out, start=_seg, duration=_rv_clip_secs,
                                      width=video_width, height=video_height, mute=True)
                        except Exception as _ce:
                            print(f"[clip_mix] trim error t{_tidx+1} c{_cidx+1}: {_ce}")
                    if _out.exists() and _rv_clip_dur(_out) > 0:
                        _rv_all_clips.append(_out)
                    _seg += _rv_clip_secs
                    _cidx += 1

            _rv_rnd.shuffle(_rv_all_clips)
            print(f"[clip_mix] {len(_rv_all_clips)} trailer clips ready (randomized)")

            # Assign clips to slides (cycle if more slides than clips).
            # First split any slides longer than clip_secs so every clip
            # fully covers its slide — no frozen last-frame padding.
            if _rv_all_clips:
                _split_slides: list = []
                for _sl in slides:
                    _sdur = _sl.end - _sl.start
                    if _sdur > _rv_clip_secs + 0.05:
                        # Split into sub-slides of at most _rv_clip_secs each
                        _t = _sl.start
                        while _sl.end - _t > 0.05:
                            _sub_end = min(_t + _rv_clip_secs, _sl.end)
                            _split_slides.append(Slide(
                                start=_t, end=_sub_end,
                                image_path=_sl.image_path, query=_sl.query,
                                headline=_sl.headline, subhead=_sl.subhead,
                                source_page=_sl.source_page, image_url=_sl.image_url,
                                license_name=_sl.license_name, license_url=_sl.license_url,
                                attribution=_sl.attribution, motion=_sl.motion,
                                window_text=_sl.window_text, window_keywords=_sl.window_keywords,
                                queries_tried=_sl.queries_tried,
                            ))
                            _t += _rv_clip_secs
                    else:
                        _split_slides.append(_sl)
                slides = _split_slides

                _ci = 0
                for _si in range(len(slides)):
                    _sl = slides[_si]
                    _cp = _rv_all_clips[_ci % len(_rv_all_clips)]
                    _ci += 1
                    _actual_dur = min(_rv_clip_dur(_cp), _sl.end - _sl.start)
                    slides[_si] = Slide(
                        start=_sl.start, end=_sl.end,
                        image_path=_sl.image_path, query=_sl.query,
                        headline=_sl.headline, subhead=_sl.subhead,
                        source_page=_sl.source_page, image_url=_sl.image_url,
                        license_name=_sl.license_name, license_url=_sl.license_url,
                        attribution=_sl.attribution, motion=_sl.motion,
                        window_text=_sl.window_text, window_keywords=_sl.window_keywords,
                        queries_tried=_sl.queries_tried,
                        video_clip_path=str(_cp),
                        video_clip_start=0.0,
                        video_clip_end=_actual_dur,
                        video_clip_mute=True,
                    )

            # Write prepared_clips.json for reuse compatibility.
            write_json(out_dir / "prepared_clips.json", [
                {"path": str(_cp), "timeline_start": 0.0, "timeline_end": _rv_clip_secs,
                 "source_url": "", "source_title": "", "search_query": f"{_clip_topic} official trailer",
                 "muted": True}
                for _cp in _rv_all_clips
            ])

            # ── Legacy reuse path kept for back-compat (never reached now) ──
            if False:
                _prior_clips_json = None  # dead code sentinel
                _prior_data: list = []
                for _pc_d in (_prior_data or []):
                    _src_p = Path(str(_pc_d.get("path", "")))
        except Exception as exc:
            # Video clip mixing is best-effort; don't fail the whole pipeline.
            import traceback
            traceback.print_exc()

    # ── Clip-review: fill EVERY slide with muted movie clips (OVERLAY, no pauses) ──
    if vt == "clip_review":
        try:
            from .clip_suggestions import suggest_full_coverage_clips
            from .clip_tools import (
                prepare_clip_for_suggestion as _prep_cr,
                get_video_duration as _clip_dur_cr,
                PreparedClip as _PC_cr,
            )

            _use_research_cr = clip_research
            if _use_research_cr:
                from .clip_researcher import research_and_prepare_clip as _prep_research_cr
                print("[pipeline] clip_review: using AGENTIC clip research")

            clip_dir = ensure_dir(out_dir / "clips")

            cr_suggestions = suggest_full_coverage_clips(
                segments=segments,
                topic=effective_topic or audio_path.stem,
                title=audio_path.stem,
                audio_duration=timeline_duration,
                target_clip_seconds=8.0,
                model=llm_model,
            )
            print(f"[pipeline] clip_review: {len(cr_suggestions)} clip segments covering {timeline_duration:.1f}s")

            write_json(out_dir / "clip_review_suggestions.json", [
                {"timeline_start": s.timeline_start, "timeline_end": s.timeline_end,
                 "search_query": s.search_query, "reason": s.reason, "mute": s.mute}
                for s in cr_suggestions
            ])

            # Download and prepare each clip.
            cr_prepared: list[_PC_cr] = []
            _seen_cr_urls: set[str] = set()
            for ci, sug in enumerate(cr_suggestions):
                try:
                    if _use_research_cr:
                        ctx_start = max(0.0, sug.timeline_start - 15)
                        ctx_end = sug.timeline_end + 15
                        ctx_lines = [
                            f"[{s.start:.1f}-{s.end:.1f}] {s.text}"
                            for s in segments
                            if s.start >= ctx_start and s.end <= ctx_end
                        ]
                        transcript_ctx = "\n".join(ctx_lines)
                        pcs = _prep_research_cr(
                            sug,
                            dest_dir=clip_dir,
                            clip_index=ci,
                            width=video_width,
                            height=video_height,
                            topic=effective_topic or audio_path.stem,
                            transcript_context=transcript_ctx,
                            llm_model="gpt-4o",
                            seen_urls=_seen_cr_urls,
                            max_scrape_pages=5,
                            max_clips=1,
                        )
                        if pcs:
                            cr_prepared.append(pcs[0])
                    else:
                        pc = _prep_cr(
                            sug,
                            dest_dir=clip_dir,
                            clip_index=ci,
                            width=video_width,
                            height=video_height,
                            llm_model=llm_model,
                            sort_by_views=True,
                            mode="review",
                            seen_urls=_seen_cr_urls,
                        )
                        if pc is not None:
                            cr_prepared.append(pc)
                except Exception:
                    import traceback; traceback.print_exc()

            print(f"[pipeline] clip_review: {len(cr_prepared)}/{len(cr_suggestions)} clips downloaded")

            # OVERLAY mode: replace every slide with the matching video clip.
            # Each prepared clip covers a timeline range — find the overlapping slides
            # and assign the clip.  Unlike commentary INSERT mode, the timeline stays
            # unchanged; narration continues uninterrupted.
            cr_prepared.sort(key=lambda p: p.timeline_start)

            # Rebuild the slide list so that every time range is covered by a clip.
            new_slides: list[Slide] = []
            for pc in cr_prepared:
                actual_dur = _clip_dur_cr(pc.path)
                if actual_dur <= 0:
                    actual_dur = pc.timeline_end - pc.timeline_start
                clip_dur = min(actual_dur, pc.timeline_end - pc.timeline_start)

                # Find an existing slide that overlaps this clip time range to
                # inherit image_path (used as fallback poster if clip fails).
                fallback_slide = slides[0] if slides else None
                for sl in slides:
                    if sl.start < pc.timeline_end and sl.end > pc.timeline_start:
                        fallback_slide = sl
                        break

                fb = fallback_slide or slides[0]
                new_slides.append(Slide(
                    start=pc.timeline_start,
                    end=pc.timeline_start + clip_dur,
                    image_path=fb.image_path, query=fb.query,
                    headline=fb.headline, subhead=fb.subhead,
                    source_page=fb.source_page, image_url=fb.image_url,
                    license_name=fb.license_name, license_url=fb.license_url,
                    attribution=fb.attribution, motion=fb.motion,
                    window_text=fb.window_text, window_keywords=fb.window_keywords,
                    queries_tried=fb.queries_tried,
                    video_clip_path=str(pc.path),
                    video_clip_start=0.0,
                    video_clip_end=clip_dur,
                    video_clip_mute=True,
                ))

            # Fill any gaps between prepared clips with image slides from the
            # original list so there's never a blank screen.
            if new_slides:
                gap_fills: list[Slide] = []
                # Gap before first clip.
                if new_slides[0].start > 0.5:
                    fb = slides[0] if slides else new_slides[0]
                    gap_fills.append(Slide(
                        start=0.0, end=new_slides[0].start,
                        image_path=fb.image_path, query=fb.query,
                        headline=fb.headline, subhead=fb.subhead,
                        source_page=fb.source_page, image_url=fb.image_url,
                        license_name=fb.license_name, license_url=fb.license_url,
                        attribution=fb.attribution, motion=fb.motion,
                        window_text=fb.window_text, window_keywords=fb.window_keywords,
                        queries_tried=fb.queries_tried,
                    ))
                # Gaps between consecutive clips.
                for i in range(len(new_slides) - 1):
                    gap_start = new_slides[i].end
                    gap_end = new_slides[i + 1].start
                    if gap_end - gap_start > 0.5:
                        fb = slides[0] if slides else new_slides[i]
                        gap_fills.append(Slide(
                            start=gap_start, end=gap_end,
                            image_path=fb.image_path, query=fb.query,
                            headline=fb.headline, subhead=fb.subhead,
                            source_page=fb.source_page, image_url=fb.image_url,
                            license_name=fb.license_name, license_url=fb.license_url,
                            attribution=fb.attribution, motion=fb.motion,
                            window_text=fb.window_text, window_keywords=fb.window_keywords,
                            queries_tried=fb.queries_tried,
                        ))
                # Gap after last clip.
                if new_slides[-1].end < timeline_duration - 0.5:
                    fb = slides[-1] if slides else new_slides[-1]
                    gap_fills.append(Slide(
                        start=new_slides[-1].end, end=timeline_duration,
                        image_path=fb.image_path, query=fb.query,
                        headline=fb.headline, subhead=fb.subhead,
                        source_page=fb.source_page, image_url=fb.image_url,
                        license_name=fb.license_name, license_url=fb.license_url,
                        attribution=fb.attribution, motion=fb.motion,
                        window_text=fb.window_text, window_keywords=fb.window_keywords,
                        queries_tried=fb.queries_tried,
                    ))
                new_slides.extend(gap_fills)
                new_slides.sort(key=lambda s: s.start)

            if new_slides:
                slides = new_slides
                print(f"[pipeline] clip_review: {len(slides)} total slides (clips + gap fills)")

            write_json(out_dir / "prepared_clips.json", [
                {"path": str(pc.path), "timeline_start": pc.timeline_start,
                 "timeline_end": pc.timeline_end, "source_url": pc.source_url,
                 "source_title": pc.source_title, "search_query": pc.search_query,
                 "muted": pc.muted}
                for pc in cr_prepared
            ])
        except Exception:
            import traceback
            traceback.print_exc()

    # ── Commentary clip insertion: always-on for commentary type,
    #    also activated when --clip-queries or --clip-research is explicitly set ──
    if vt == "commentary" or clip_queries or clip_research:
        # Commentary always uses the commentary-specific LLM to find reference clips
        # and build compilation montages.
        # If the user supplied manual `clip_queries`, those REPLACE the LLM-generated
        # search queries.  The LLM still determines WHERE to insert clips in the
        # timeline, but the *what* is controlled by the user.
        try:
            from .commentary_clips import suggest_commentary_clips
            from .clip_tools import (
                prepare_clip_for_suggestion as _prep_ref,
                prepare_compilation_clip as _prep_comp,
                get_video_duration as _clip_dur,
                PreparedClip,
            )
            from .models import VideoClipSuggestion as _VCS, CommentaryClipSuggestion as _CCS

            # Agentic research pipeline for finding clips.
            _use_research = clip_research
            if _use_research:
                from .clip_researcher import research_and_prepare_clip as _prep_research
                print("[pipeline] Using AGENTIC clip research pipeline")

            clip_dir = ensure_dir(out_dir / "clips")

            if clip_queries:
                # ── Manual clip queries: group ALL as a single compilation ──
                # Scan transcript for explicit cue phrases first — much more
                # reliable than asking the LLM for just 1 insertion point.
                import re as _re
                _CUE_RE = _re.compile(
                    r"(?:let'?s\s+(?:take\s+a\s+)?look|let'?s\s+(?:watch|see)|"
                    r"here'?s\s+the\s+clip|watch\s+this|check\s+(?:this|it)\s+out|"
                    r"play\s+the\s+clip|roll\s+the\s+clip|look\s+at\s+(?:this|the)|"
                    r"let\s+me\s+show\s+you|take\s+a\s+look)",
                    _re.IGNORECASE,
                )
                cue_start: float | None = None
                for seg in segments:
                    if _CUE_RE.search(seg.text):
                        cue_start = seg.start
                        print(f"[pipeline] Found cue phrase at {cue_start:.1f}s: {seg.text.strip()[:80]}")
                        break  # use first cue phrase

                if cue_start is not None:
                    insert_start = cue_start
                else:
                    # Fallback: ask LLM to find the insertion point.
                    csug = suggest_commentary_clips(
                        segments=segments,
                        topic=effective_topic or audio_path.stem,
                        title=audio_path.stem,
                        audio_duration=timeline_duration,
                        max_clips=1,
                        model=llm_model,
                    )
                    if csug:
                        insert_start = csug[0].timeline_start
                    else:
                        insert_start = round(timeline_duration * 0.15, 1)
                    print(f"[pipeline] No cue phrase found, using LLM/default insertion at {insert_start:.1f}s")

                # Enrich short queries with topic context.
                _raw_topic = (effective_topic or audio_path.stem or "").strip()
                _first_phrase = _raw_topic.split(",")[0].strip()
                topic_ctx = " ".join(_first_phrase.split()[:6])

                enriched: list[str] = []
                for q in clip_queries:
                    eq = q.strip()
                    if topic_ctx and len(eq.split()) <= 4 and topic_ctx.lower() not in eq.lower():
                        eq = f"{eq} {topic_ctx}"
                    enriched.append(eq)

                # Build a single compilation suggestion with all queries.
                primary_query = enriched[0]
                extra = enriched[1:] if len(enriched) > 1 else None
                dur = 15.0 * len(enriched)  # ~15s per clip
                csug = [_CCS(
                    timeline_start=insert_start,
                    timeline_end=round(insert_start + dur, 1),
                    search_query=primary_query,
                    reason=f"user-specified compilation: {', '.join(clip_queries)}",
                    clip_type="compilation",
                    num_clips=len(enriched),
                    extra_queries=extra,
                    mute=False,
                )]

            else:
                # ── Fully LLM-driven clip detection ──
                csug = suggest_commentary_clips(
                    segments=segments,
                    topic=effective_topic or audio_path.stem,
                    title=audio_path.stem,
                    audio_duration=timeline_duration,
                    max_clips=max_video_clips,
                    model=llm_model,
                )

            write_json(out_dir / "commentary_clip_suggestions.json", [
                {"timeline_start": s.timeline_start, "timeline_end": s.timeline_end,
                 "search_query": s.search_query, "reason": s.reason,
                 "clip_type": s.clip_type, "num_clips": s.num_clips,
                 "extra_queries": s.extra_queries, "mute": s.mute}
                for s in csug
            ])

            # Build transcript context for the research agent.
            _transcript_text = "\n".join(f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in segments)

            prepared_clips: list[PreparedClip] = []
            _seen_clip_urls: set[str] = set()  # avoid downloading same video for different queries
            _clip_sub_idx = 0  # running counter for unique clip file names
            for ci, sug in enumerate(csug):
                try:
                    if _use_research:
                        # ── Agentic research pipeline ──
                        # Extract transcript window around the clip for context.
                        ctx_start = max(0.0, sug.timeline_start - 30)
                        ctx_end = sug.timeline_end + 30
                        ctx_lines = [
                            f"[{s.start:.1f}-{s.end:.1f}] {s.text}"
                            for s in segments
                            if s.start >= ctx_start and s.end <= ctx_end
                        ]
                        transcript_ctx = "\n".join(ctx_lines)
                        # research_and_prepare_clip now returns a *list* of PreparedClip
                        pcs = _prep_research(
                            sug,
                            dest_dir=clip_dir,
                            clip_index=_clip_sub_idx,
                            width=video_width,
                            height=video_height,
                            topic=effective_topic or audio_path.stem,
                            transcript_context=transcript_ctx,
                            llm_model="gpt-4o",
                            seen_urls=_seen_clip_urls,
                            max_scrape_pages=5,
                            max_clips=max(1, int(getattr(sug, "num_clips", 1))),
                        )
                        if pcs:
                            prepared_clips.extend(pcs)
                            _clip_sub_idx += len(pcs)
                    elif sug.clip_type == "compilation":
                        pc = _prep_comp(
                            sug,
                            dest_dir=clip_dir,
                            clip_index=ci,
                            width=video_width,
                            height=video_height,
                            llm_model=llm_model,
                            sort_by_views=True,
                            mode="commentary",
                        )
                        if pc is not None:
                            prepared_clips.append(pc)
                    else:
                        # Reference clip: use existing single-clip flow.
                        ref_sug = _VCS(
                            timeline_start=sug.timeline_start,
                            timeline_end=sug.timeline_end,
                            search_query=sug.search_query,
                            reason=sug.reason,
                            mute=sug.mute,
                        )
                        pc = _prep_ref(
                            ref_sug,
                            dest_dir=clip_dir,
                            clip_index=ci,
                            width=video_width,
                            height=video_height,
                            llm_model=llm_model,
                            sort_by_views=False,
                            mode="commentary",
                            seen_urls=_seen_clip_urls,
                        )
                        if pc is not None:
                            prepared_clips.append(pc)
                except Exception:
                    pass

            # ── INSERT mode: clips pause narration and extend the video ──
            # Sort clips by their original insertion point so we can process
            # them in order and accumulate the total timeline shift.
            prepared_clips.sort(key=lambda p: p.timeline_start)

            # Group clips that share the same insertion point (e.g., multiple
            # reaction clips for one commentary suggestion).
            from itertools import groupby as _groupby

            clip_groups: list[tuple[float, list[PreparedClip]]] = []
            for _tls, grp in _groupby(prepared_clips, key=lambda p: p.timeline_start):
                clip_groups.append((float(_tls), list(grp)))

            total_shift = 0.0  # accumulated timeline extension so far

            for insert_at_orig, group in clip_groups:
                # Compute each clip's actual on-disk duration.
                clip_durations: list[float] = []
                for pc in group:
                    d = _clip_dur(pc.path)
                    if d <= 0:
                        d = pc.timeline_end - pc.timeline_start
                    clip_durations.append(max(0.5, d))
                group_dur = sum(clip_durations)

                # The narration time at which to insert silence (original audio time).
                audio_insert_time = insert_at_orig
                audio_pause_insertions.append((audio_insert_time, group_dur))

                # The video-timeline position (accounting for previous insertions).
                insert_at_video = insert_at_orig + total_shift

                # Find the slide that contains the insertion point and split it.
                best_idx = -1
                for si, sl in enumerate(slides):
                    if sl.video_clip_path:
                        continue
                    if sl.start <= insert_at_video < sl.end:
                        best_idx = si
                        break
                if best_idx < 0:
                    # Fallback: insert after the last slide before the insertion point.
                    for si, sl in enumerate(slides):
                        if sl.start <= insert_at_video:
                            best_idx = si
                    if best_idx < 0:
                        best_idx = 0

                sl = slides[best_idx]
                new_slides: list[Slide] = []

                # Part A: image before the clip insertion point.
                if insert_at_video - sl.start > 0.2:
                    new_slides.append(Slide(
                        start=sl.start, end=insert_at_video,
                        image_path=sl.image_path, query=sl.query,
                        headline=sl.headline, subhead=sl.subhead,
                        source_page=sl.source_page, image_url=sl.image_url,
                        license_name=sl.license_name, license_url=sl.license_url,
                        attribution=sl.attribution, motion=sl.motion,
                        window_text=sl.window_text, window_keywords=sl.window_keywords,
                        queries_tried=sl.queries_tried,
                    ))

                # Part B: each clip slide plays sequentially.
                cursor = insert_at_video
                for pc, cd in zip(group, clip_durations):
                    new_slides.append(Slide(
                        start=cursor, end=cursor + cd,
                        image_path=sl.image_path, query=sl.query,
                        headline=sl.headline, subhead=sl.subhead,
                        source_page=sl.source_page, image_url=sl.image_url,
                        license_name=sl.license_name, license_url=sl.license_url,
                        attribution=sl.attribution, motion=sl.motion,
                        window_text=sl.window_text, window_keywords=sl.window_keywords,
                        queries_tried=sl.queries_tried,
                        video_clip_path=str(pc.path),
                        video_clip_start=0.0,
                        video_clip_end=cd,
                        video_clip_mute=pc.muted,
                    ))
                    cursor += cd

                # Part C: remainder of the split slide, shifted forward.
                # When the insertion point is before this slide (no slide
                # covers that time), preserve the full original slide duration
                # instead of the nonsensical sl.end - insert_at_video.
                if insert_at_video >= sl.start:
                    remainder = sl.end - insert_at_video
                else:
                    remainder = sl.end - sl.start
                if remainder > 0.2:
                    new_slides.append(Slide(
                        start=cursor, end=cursor + remainder,
                        image_path=sl.image_path, query=sl.query,
                        headline=sl.headline, subhead=sl.subhead,
                        source_page=sl.source_page, image_url=sl.image_url,
                        license_name=sl.license_name, license_url=sl.license_url,
                        attribution=sl.attribution, motion=sl.motion,
                        window_text=sl.window_text, window_keywords=sl.window_keywords,
                        queries_tried=sl.queries_tried,
                    ))

                slides[best_idx:best_idx + 1] = new_slides

                # Shift ALL subsequent slides (those after the insertion splice)
                # forward by the clip group duration.
                splice_end_idx = best_idx + len(new_slides)
                for si in range(splice_end_idx, len(slides)):
                    s = slides[si]
                    slides[si] = Slide(
                        start=s.start + group_dur, end=s.end + group_dur,
                        image_path=s.image_path, query=s.query,
                        headline=s.headline, subhead=s.subhead,
                        source_page=s.source_page, image_url=s.image_url,
                        license_name=s.license_name, license_url=s.license_url,
                        attribution=s.attribution, motion=s.motion,
                        window_text=s.window_text, window_keywords=s.window_keywords,
                        queries_tried=s.queries_tried,
                        video_clip_path=s.video_clip_path,
                        video_clip_start=s.video_clip_start,
                        video_clip_end=s.video_clip_end,
                        video_clip_mute=s.video_clip_mute,
                    )

                total_shift += group_dur
                timeline_duration += group_dur

            if total_shift > 0:
                print(f"[pipeline] INSERT mode: video extended by {total_shift:.1f}s for clip pauses")

            write_json(out_dir / "prepared_clips.json", [
                {"path": str(pc.path), "timeline_start": pc.timeline_start,
                 "timeline_end": pc.timeline_end, "source_url": pc.source_url,
                 "source_title": pc.source_title, "search_query": pc.search_query,
                 "muted": pc.muted}
                for pc in prepared_clips
            ])
        except Exception:
            import traceback
            traceback.print_exc()

    # If we inserted clip pauses or pivot pauses, generate a padded audio file for rendering.
    if audio_pause_insertions:
        try:
            padded = _ensure_audio_with_pauses(audio_in=audio_path, insertions=audio_pause_insertions)
            if padded is not None and padded.exists() and padded.stat().st_size > 4096:
                render_audio_override = padded
                print(f"[pipeline] Padded audio with {len(audio_pause_insertions)} pause(s) → {padded.name}")
        except Exception:
            render_audio_override = None

    # Ensure the slideshow covers the full timeline duration contiguously.
    slides.sort(key=lambda s: s.start)
    stitched: list[Slide] = []
    cur = 0.0
    for idx, s in enumerate(slides):
        start = max(cur, float(s.start))
        # Provisional end; will be overwritten for all but last.
        end = max(start + 0.1, float(s.end))
        stitched.append(
            Slide(
                start=start,
                end=end,
                image_path=s.image_path,
                query=s.query,
                headline=s.headline,
                subhead=s.subhead,
                source_page=s.source_page,
                image_url=s.image_url,
                license_name=s.license_name,
                license_url=s.license_url,
                attribution=s.attribution,
                motion=s.motion,
                window_text=s.window_text,
                window_keywords=s.window_keywords,
                queries_tried=s.queries_tried,
                video_clip_path=s.video_clip_path,
                video_clip_start=s.video_clip_start,
                video_clip_end=s.video_clip_end,
                video_clip_mute=s.video_clip_mute,
            )
        )
        cur = end

    # Make ends align to next start, filling any gaps.
    for i in range(len(stitched) - 1):
        stitched[i] = Slide(
            start=stitched[i].start,
            end=max(stitched[i].start + 0.1, stitched[i + 1].start),
            image_path=stitched[i].image_path,
            query=stitched[i].query,
            headline=stitched[i].headline,
            subhead=stitched[i].subhead,
            source_page=stitched[i].source_page,
            image_url=stitched[i].image_url,
            license_name=stitched[i].license_name,
            license_url=stitched[i].license_url,
            attribution=stitched[i].attribution,
            motion=stitched[i].motion,
            window_text=stitched[i].window_text,
            window_keywords=stitched[i].window_keywords,
            queries_tried=stitched[i].queries_tried,
            video_clip_path=stitched[i].video_clip_path,
            video_clip_start=stitched[i].video_clip_start,
            video_clip_end=stitched[i].video_clip_end,
            video_clip_mute=stitched[i].video_clip_mute,
        )

    stitched[-1] = Slide(
        start=stitched[-1].start,
        end=float(timeline_duration),
        image_path=stitched[-1].image_path,
        query=stitched[-1].query,
        headline=stitched[-1].headline,
        subhead=stitched[-1].subhead,
        source_page=stitched[-1].source_page,
        image_url=stitched[-1].image_url,
        license_name=stitched[-1].license_name,
        license_url=stitched[-1].license_url,
        attribution=stitched[-1].attribution,
        motion=stitched[-1].motion,
        window_text=stitched[-1].window_text,
        window_keywords=stitched[-1].window_keywords,
        queries_tried=stitched[-1].queries_tried,
        video_clip_path=stitched[-1].video_clip_path,
        video_clip_start=stitched[-1].video_clip_start,
        video_clip_end=stitched[-1].video_clip_end,
        video_clip_mute=stitched[-1].video_clip_mute,
    )

    slides = stitched

    if slides[0].start > 0:
        # Find the first non-clip slide to use its image for the gap-fill.
        gap_src = slides[0]
        for _gs in slides:
            if not _gs.video_clip_path:
                gap_src = _gs
                break
        slides = [
            Slide(
                start=0.0,
                end=float(slides[0].start),
                image_path=gap_src.image_path,
                query=gap_src.query,
                headline=gap_src.headline,
                subhead=gap_src.subhead,
                source_page=gap_src.source_page,
                image_url=gap_src.image_url,
                license_name=gap_src.license_name,
                license_url=gap_src.license_url,
                attribution=gap_src.attribution,
                motion=gap_src.motion,
                window_text=gap_src.window_text,
                window_keywords=gap_src.window_keywords,
                queries_tried=gap_src.queries_tried,
                video_clip_path=None,
                video_clip_start=None,
                video_clip_end=None,
                video_clip_mute=False,
            )
        ] + slides

    write_json(out_dir / "timeline.json", [asdict(s) for s in slides])

    # Attribution: if reused, we may have already copied attribution.txt above.
    if not (out_dir / "attribution.txt").exists():
        (out_dir / "attribution.txt").write_text("\n".join(attribution_lines) + "\n", encoding="utf-8")

    # Minimal run metadata to help future reuse/matching.
    try:
        write_json(
            out_dir / "run_meta.json",
            {
                "audio_name": audio_path.name,
                "audio_stem": audio_path.stem,
                "audio_size": int(audio_stat.st_size),
                "audio_mtime": float(audio_stat.st_mtime),
                "render_audio_path": (str(render_audio_override.as_posix()) if render_audio_override else None),
                "video_type": vt,
                "video_width": int(video_width),
                "video_height": int(video_height),
                "topic": effective_topic,
                "topic_type": topic_type,
                "image_provider": image_provider,
                "shorts_review_max_beats": (int(shorts_review_max_beats) if (vt == "shorts_review" and shorts_review_max_beats is not None) else None),
                "max_images": (int(max_images) if vt != "shorts_review" else None),
                "min_image_width": int(min_image_width),
                "reused_from": str(reuse_source_dir) if reuse_source_dir else None,
                "created_at": time.time(),
                "cwd": os.getcwd(),
            },
        )
    except Exception:
        pass

    return out_dir


# ---------------------------------------------------------------------------
# Scripted Short — user provides a timestamped script + audio; we search
# one image per section and output a timeline.json (portrait 9:16).
# ---------------------------------------------------------------------------


def fetch_review_image_pool(
    topic: str,
    out_dir: str | Path,
    *,
    max_results: int = 25,
    min_image_width: int = 900,
    image_provider: str = "google_images",
    serpapi_api_key: str | None = None,
    progress_cb=None,
) -> list[Path]:
    """Fetch a pool of images for *topic* and return the downloaded paths.

    Called by the UI before the main pipeline so the user can pick which
    images to keep.  Images are saved into *out_dir*/assets/ with the
    ``pool##_`` prefix.
    """
    from .playwright_images import playwright_google_image_search

    out_dir = Path(out_dir)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Search
    try:
        candidates = playwright_google_image_search(
            topic, max_results=max_results, min_width=min_image_width,
        )
    except Exception as e:
        print(f"[fetch_pool] search failed: {e}")
        candidates = []

    # Download
    downloaded: list[Path] = []
    seen_hashes: set[str] = set()
    for idx, cand in enumerate(candidates[:max_results]):
        if progress_cb:
            progress_cb(idx, len(candidates[:max_results]))
        img_url = cand.get("url") or cand.get("image_url") or cand.get("img_url")
        if not img_url:
            continue
        try:
            from .playwright_images import _download_image_bytes
            raw = _download_image_bytes(img_url, min_width=min_image_width)
        except Exception:
            try:
                import urllib.request
                with urllib.request.urlopen(img_url, timeout=10) as resp:
                    raw = resp.read()
            except Exception:
                continue
        if not raw or len(raw) < 5000:
            continue
        h = hashlib.md5(raw).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        ext = ".jpg"
        if raw[:4] == b"\x89PNG":
            ext = ".png"
        dst = assets_dir / f"pool{len(downloaded):02d}_{int(time.time() * 1000)}{ext}"
        dst.write_bytes(raw)
        downloaded.append(dst)

    print(f"[fetch_pool] Downloaded {len(downloaded)} images for {topic!r}")
    return downloaded


def run_scripted_short(
    *,
    audio_path: str | Path,
    out_dir: str | Path,
    script_sections: list,  # list[ScriptSection] from script_parser
    topic: str | None = None,
    image_provider: str = "google_images",
    serpapi_api_key: str | None = None,
    min_image_width: int = 900,
    video_width: int = 1080,
    video_height: int = 1920,
    llm_model: str = "gpt-4o-mini",
    user_images: list[dict] | None = None,
    reuse_images: bool = True,
    use_clips: bool = False,
) -> Path:
    """Build a timeline for a **Scripted Short**.

    The *script_sections* drive image search queries and on-screen text
    overlays.  Slide timing is derived from Whisper word-level timestamps
    so each image appears when the narrator reaches its section.

    *user_images* is an optional list of ``{"title": str, "path": str}``
    dicts.  When a title semantically matches a section (label or
    narration), that image is used directly — no search needed.

    Pipeline:
    1. Matches user-provided reference images to sections by title.
    2. Uses the LLM to generate an ideal image-search query per section.
    3. Searches, ranks, and downloads one image per section.
    4. Applies portrait pre-processing (blur-behind composite).
    5. Syncs slide timing to audio via word-level timestamps.
    6. Writes ``timeline.json`` and ``run_meta.json``.

    Returns *out_dir*.
    """

    from .script_parser import ScriptSection  # type-check convenience

    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = ensure_dir(out_dir / "assets")

    audio_stat = audio_path.stat()
    audio_duration = get_audio_duration_seconds(audio_path)

    sections: list[ScriptSection] = list(script_sections)
    if not sections:
        raise ValueError("script_sections is empty")

    effective_topic = (topic or "").strip() or None

    # ── Shorten topic for better image search queries ──
    # Verbose topics like "True story of backrooms" produce poor queries.
    # Strip common filler prefixes to get the core subject.
    _TOPIC_FILLER_PREFIXES = [
        "the true story of", "true story of", "the story of", "story of",
        "the history of", "history of", "the real story of", "real story of",
        "the truth about", "truth about", "everything about",
        "a look at", "an introduction to", "introduction to",
        "the rise of", "rise of", "the fall of", "fall of",
        "the mystery of", "mystery of", "the secret of", "secret of",
        "what is", "what are", "who is", "who are",
        "how the", "why the", "the best of", "best of",
    ]
    if effective_topic:
        t_lower = effective_topic.lower().strip()
        for prefix in _TOPIC_FILLER_PREFIXES:
            if t_lower.startswith(prefix):
                remainder = effective_topic[len(prefix):].strip()
                if remainder:
                    effective_topic = remainder
                break

    # ── 1. Generate search queries per section ──
    api_key = os.getenv("OPENAI_API_KEY")

    # ── Short, punchy transcript-based queries ──
    # Goal: 2-4 word Google image searches like "sinners vampires",
    # "smoke stack sinners", "twins sinners" — not long sentence fragments.
    import re as _re

    # Derive a short "slug" from the topic for appending to every query.
    # e.g. "The Sinners 2025 movie" → "sinners"
    #      "Bad Bunny Halftime Show" → "bad bunny"
    #      "Backrooms explained" → "backrooms"
    _TOPIC_DROP = _re.compile(
        r"\b(movie|film|show|series|tv|explained|theories|theory|season|episode"
        r"|part|20\d\d|the|a|an)\b",
        _re.IGNORECASE,
    )
    _topic_slug = ""
    if effective_topic:
        _slug_words = _TOPIC_DROP.sub("", effective_topic).split()
        _topic_slug = " ".join(_slug_words[:2]).strip().lower()  # max 2 words

    # Stopwords that add no search value in an OSD label
    _OSD_NOISE = {
        "as", "the", "a", "an", "of", "in", "to", "and", "or", "for",
        "is", "are", "its", "vs", "vs.", "it", "this", "that", "why",
        "does", "not", "actually", "real", "true", "two", "three",
        "machine", "language", "strategy", "strategies",
        # additional filler words
        "keep", "promise", "alternate", "selves", "survival",
        "colonizers", "assimilation", "competing", "internal",
        "literal", "alternate", "version", "versions",
    }

    def _osd_to_keywords(osd_text: str, max_kw: int = 2) -> str:
        """Strip numbering + noise words; return top N content words."""
        # Remove leading "1)" / "Number 1:" etc.
        text = _re.sub(
            r"^(?:number\s*)?\d+[\.\:\)\-]?\s*", "", osd_text, flags=_re.IGNORECASE
        )
        # Remove punctuation except letters/digits/spaces
        text = _re.sub(r"[^\w\s]", " ", text)
        words = [w.lower() for w in text.split() if w.lower() not in _OSD_NOISE and len(w) >= 3]
        return " ".join(words[:max_kw])

    for i, s in enumerate(sections):
        label = (s.label or "").strip()
        osd   = (s.on_screen_text or "").strip() if hasattr(s, "on_screen_text") else ""

        # INTRO → just the topic slug (e.g. "sinners")
        if label.upper() == "INTRO":
            s.search_query = _topic_slug or effective_topic or label
            print(f"  [query s{i:02d}] INTRO → {s.search_query!r}")
            continue

        # Core: key words from OSD label (most concise description of section)
        core = _osd_to_keywords(osd or label, max_kw=3)

        # If OSD gave nothing useful, fall back to top keywords from narration
        if not core:
            core = " ".join(extract_keywords(s.text, max_words=3))

        # Final query: "[core concept] [topic slug]" — short and specific
        if _topic_slug and _topic_slug not in core:
            query = f"{core} {_topic_slug}".strip()
        else:
            query = core.strip() or _topic_slug or effective_topic or label

        s.search_query = query or "background"
        print(f"  [query s{i:02d}] osd={osd!r}  → {s.search_query!r}")

    print(f"[scripted_short] Transcript-based queries generated")

    # Fallback: derive queries from section text content.
    # Extract the most distinctive terms from each section's narration.
    for i, s in enumerate(sections):
        if not s.search_query:
            kws = extract_keywords(s.text, max_words=5)
            if effective_topic:
                # Combine topic + section-specific keywords.
                kw_str = " ".join(kws[:4]).strip()
                if kw_str and effective_topic.lower() not in kw_str.lower():
                    s.search_query = f"{effective_topic} {kw_str}"
                elif kw_str:
                    s.search_query = kw_str
                else:
                    s.search_query = f"{effective_topic} {s.label}"
            else:
                s.search_query = " ".join(kws[:5]).strip() or s.label or "background"

    print(f"[scripted_short] {len(sections)} sections, queries:")
    for i, s in enumerate(sections):
        print(f"  [{i}] {s.start:.1f}-{s.end:.1f} ({s.label}): {s.search_query}")

    # ── 1b. Match user-provided reference images to sections ──
    # Maps section index → local file path (pre-assigned, skip search).
    _user_image_map: dict[int, str] = {}
    if user_images:
        _ui_items = [ui for ui in user_images if ui.get("title") and ui.get("path") and Path(ui["path"]).exists()]
        if _ui_items and api_key:
            # Use LLM to semantically match titles to sections.
            try:
                from .llm_storyboard import _openai_chat_completions
                import json as _json

                _sec_descs = []
                for idx, s in enumerate(sections):
                    _sec_descs.append(f"Section {idx} \"{s.label}\": {s.text[:200]}")
                _img_descs = []
                for idx, ui in enumerate(_ui_items):
                    _img_descs.append(f"Image {idx}: \"{ui['title']}\"")

                _match_sys = (
                    "You match user-provided reference images to video script sections.\n"
                    "Each image has a descriptive title. Match it to the section whose "
                    "content it best illustrates.\n\n"
                    "Return ONLY valid JSON: {\"matches\": [{\"image\": 0, \"section\": 2}, ...]}\n"
                    "Only include confident matches. If an image title doesn't clearly "
                    "match any section, omit it.\n"
                )
                _match_user = (
                    "Sections:\n" + "\n".join(_sec_descs) + "\n\n"
                    "Images:\n" + "\n".join(_img_descs) + "\n\n"
                    "Match each image to the most relevant section."
                )
                _match_resp = _openai_chat_completions(
                    api_key=api_key,
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": _match_sys},
                        {"role": "user", "content": _match_user},
                    ],
                )
                import re as _re
                _ms = _match_resp.strip()
                _mf = _re.search(r"```(?:json)?\s*\n?(.*?)```", _ms, _re.DOTALL)
                if _mf:
                    _ms = _mf.group(1).strip()
                _parsed_matches = _json.loads(_ms)
                _m_list = _parsed_matches.get("matches", []) if isinstance(_parsed_matches, dict) else []
                for _m in _m_list:
                    _img_idx = int(_m.get("image", -1))
                    _sec_idx = int(_m.get("section", -1))
                    if 0 <= _img_idx < len(_ui_items) and 0 <= _sec_idx < len(sections):
                        _user_image_map[_sec_idx] = _ui_items[_img_idx]["path"]
                        print(f"  [user image] Section {_sec_idx} ({sections[_sec_idx].label}) ← \"{_ui_items[_img_idx]['title']}\"")
            except Exception as _e:
                print(f"[scripted_short] LLM image-title matching failed: {_e}")

        if not _user_image_map and _ui_items:
            # Fallback: simple keyword overlap matching (no LLM).
            for ui in _ui_items:
                title_words = set(ui["title"].lower().split())
                best_idx, best_score = -1, 0
                for idx, s in enumerate(sections):
                    if idx in _user_image_map:
                        continue
                    section_words = set((s.label + " " + s.text[:200]).lower().split())
                    overlap = len(title_words & section_words)
                    if overlap > best_score:
                        best_score = overlap
                        best_idx = idx
                if best_idx >= 0 and best_score >= 1:
                    _user_image_map[best_idx] = ui["path"]
                    print(f"  [user image] Section {best_idx} ({sections[best_idx].label}) ← \"{ui['title']}\" (keyword match)")

        if _user_image_map:
            print(f"[scripted_short] {len(_user_image_map)} section(s) pre-assigned from user images")
        elif user_images:
            print(f"[scripted_short] No user images matched any section")

    # ── 2. Image search helpers (same logic as main pipeline) ──
    _is_portrait = int(video_height) > int(video_width) * 1.1

    used_image_hashes: set[str] = set()
    used_source_pages: list[str] = []

    def _hash_file(p: str) -> str:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def _search(qstr: str) -> list[dict]:
        """Search images using Playwright (primary) with SerpAPI/DuckDuckGo fallback."""
        if not qstr:
            return []

        # Primary: Playwright-based Google Images scraping (free, no API key).
        try:
            from .playwright_images import playwright_google_image_search
            results = playwright_google_image_search(
                qstr, max_results=25, min_width=min_image_width,
            )
            if results:
                return results
        except Exception as e:
            print(f"[scripted_short] Playwright image search failed: {e}")

        # Fallback: SerpAPI if available.
        if image_provider == "google_images" and serpapi_api_key:
            try:
                from .serpapi_provider import search_google_images_candidates_via_serpapi
                return search_google_images_candidates_via_serpapi(
                    qstr, api_key=serpapi_api_key,
                    min_width=min_image_width, max_results=25,
                )
            except Exception as e:
                print(f"[scripted_short] SerpAPI fallback failed: {e}")

        # Last resort: DuckDuckGo.
        try:
            from .playwright_images import _ddg_image_fallback
            return _ddg_image_fallback(qstr, max_results=25, min_width=min_image_width)
        except Exception:
            pass

        return []

    def _download_unique(info: dict, *, prefix: str) -> str | None:
        dl_url = info.get("original_url") or info.get("image_url")
        if not dl_url:
            return None
        p = download_image(dl_url, assets_dir, prefix=prefix)
        h = _hash_file(str(p))
        if h in used_image_hashes:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
            return None
        used_image_hashes.add(h)
        return str(p)

    # ── Poster / bad-image filters ──

    _POSTER_KWS = frozenset((
        "poster", "cover", "artwork", "official-art", "keyart",
        "key-art", "dvd", "bluray", "blu-ray", "boxart", "flyer",
        "logo", "wordmark", "merchandise", "tshirt", "t-shirt",
        "wallpaper", "fan-art", "fanart", "album", "soundtrack",
        "theatrical", "one-sheet", "teaser_poster", "banner",
    ))

    def _is_bad_candidate(cand: dict) -> bool:
        """Return True if candidate should be rejected outright."""
        _url = (cand.get("original_url") or cand.get("image_url") or "").lower()
        _title = (cand.get("title") or "").lower()
        _page = (cand.get("page_url") or "").lower()
        _blob = f"{_url} {_title} {_page}"

        if any(kw in _blob for kw in _POSTER_KWS):
            return True
        if "/posters/" in _url or "/covers/" in _url:
            return True
        if "image.tmdb.org/t/p/" in _url:
            return True
        if "imdb.com" in _url and ("mediaviewer" in _page or "_V1_" in _url):
            _w = cand.get("width") or 0
            _h = cand.get("height") or 1
            if _w and _h and (_h / max(1, _w)) > 1.2:
                return True
        # Very tall images are almost certainly posters.
        _w = cand.get("width") or 0
        _h = cand.get("height") or 0
        if _w and _h and _h > 0 and (_h / max(1, _w)) > 1.6:
            return True
        return False

    # ── LLM agent panel for image selection ──

    def _agent_pick_image(
        candidates: list[dict],
        *,
        section_label: str,
        section_text: str,
        query: str,
        topic: str,
    ) -> int:
        """Use a panel of LLM 'agents' to discuss and select the best image.

        Three perspectives debate which candidate is best:
          - Visual Director: wants the most cinematic, specific scene still
          - Fact Checker: ensures the image matches the actual content
          - Anti-Poster Agent: vetoes anything that looks like a poster/promo

        Returns the index into *candidates* of the chosen image.
        """
        import json as _json
        from .llm_storyboard import _openai_chat_completions
        _api_key = os.getenv("OPENAI_API_KEY")
        if not _api_key or not candidates:
            return 0  # fallback to first candidate

        compact = []
        for ci, c in enumerate(candidates[:12]):
            compact.append({
                "i": ci,
                "title": (c.get("title") or "")[:120],
                "source": (c.get("page_url") or "")[:120],
                "url": (c.get("original_url") or c.get("image_url") or "")[:120],
                "w": int(c.get("width") or 0),
                "h": int(c.get("height") or 0),
            })

        system = (
            "You are a panel of three expert agents selecting the BEST image for a "
            "YouTube Short video slide. You must debate and agree on ONE image.\n\n"
            "AGENT 1 — Visual Director:\n"
            "  Wants a real PHOTOGRAPH or movie/TV SCENE STILL showing a recognizable "
            "  person or moment. Prefers close-ups of actors in character, dramatic shots, "
            "  candid moments. Hates generic stock photos.\n\n"
            "AGENT 2 — Fact Checker:\n"
            "  Ensures the image actually matches the section topic. The title, URL, and "
            "  source page give clues. If the title mentions a different movie/show/person, "
            "  reject it. Prefers images from reputable entertainment sites.\n\n"
            "AGENT 3 — Anti-Poster Agent:\n"
            "  VETOES any image that is likely a movie poster, DVD cover, promotional art, "
            "  logo, text-heavy graphic, fan art, or merchandise. Clues: tall aspect ratio "
            "  (h > w × 1.3), 'poster'/'cover'/'artwork' in title/URL, TMDB/IMDB poster URLs, "
            "  stock photo sites (shutterstock, alamy, getty). Also vetoes images from "
            "  Pinterest, Etsy, Redbubble, Amazon product pages.\n\n"
            "PROCESS:\n"
            "1. Each agent briefly evaluates the candidates (1 sentence each)\n"
            "2. They discuss disagreements\n"
            "3. They agree on the best index\n\n"
            "Return ONLY valid JSON (no markdown):\n"
            '{"discussion": "brief 2-3 sentence summary of the debate", '
            '"chosen_index": <int>}'
        )

        user = _json.dumps({
            "topic": topic,
            "section_label": section_label,
            "section_narration": section_text[:300],
            "search_query": query,
            "candidates": compact,
        }, ensure_ascii=False)

        try:
            content = _openai_chat_completions(
                api_key=_api_key,
                model=llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                timeout_s=25,
            )
            stripped = content.strip()
            fence = _re_mod.search(r"```(?:json)?\s*\n?(.*?)```", stripped, _re_mod.DOTALL)
            if fence:
                stripped = fence.group(1).strip()
            result = _json.loads(stripped)
            idx = int(result.get("chosen_index", 0))
            discussion = result.get("discussion", "")
            if discussion:
                print(f"    [agents] {discussion[:120]}")
            if 0 <= idx < len(candidates):
                return idx
        except Exception as e:
            print(f"    [agents] LLM pick failed ({e}), using first candidate")
        return 0

    import re as _re_mod

    def _preprocess_portrait(src: str, *, prefix: str) -> str:
        if not _is_portrait:
            return src
        try:
            from PIL import Image, ImageEnhance, ImageFilter

            img = Image.open(src).convert("RGB")
            sw, sh = img.size
            if sw <= 0 or sh <= 0:
                return src
            ratio = sw / sh
            tgt_ratio = video_width / max(1, video_height)
            if ratio / max(0.01, tgt_ratio) < 1.4:
                # Aspect ratio already close to target — no blur-behind
                # composite needed.  Still resize to exact target dims
                # and save as JPEG so ALL concat inputs share the same
                # codec (avoids PNG-vs-JPEG mismatch in FFmpeg concat).
                tw, th = video_width, video_height
                fg_scale = min(tw / sw, th / sh)
                resized = img.resize(
                    (max(1, int(sw * fg_scale + 0.5)),
                     max(1, int(sh * fg_scale + 0.5))),
                    Image.LANCZOS,
                )
                canvas = Image.new("RGB", (tw, th), (0, 0, 0))
                fx = (tw - resized.width) // 2
                fy = (th - resized.height) // 2
                canvas.paste(resized, (fx, fy))
                out = str(assets_dir / f"{prefix}_portrait.jpg")
                canvas.save(out, quality=95)
                return out
            tw, th = video_width, video_height
            bg_scale = max(tw / sw, th / sh)
            bg = img.resize((int(sw * bg_scale + 0.5), int(sh * bg_scale + 0.5)), Image.LANCZOS)
            bx, by = (bg.width - tw) // 2, (bg.height - th) // 2
            bg = bg.crop((bx, by, bx + tw, by + th))
            bg = bg.filter(ImageFilter.GaussianBlur(radius=30))
            bg = ImageEnhance.Brightness(bg).enhance(0.35)
            fg_scale = min(tw / sw, th / sh)
            fg = img.resize((int(sw * fg_scale + 0.5), int(sh * fg_scale + 0.5)), Image.LANCZOS)
            fx, fy = (tw - fg.width) // 2, (th - fg.height) // 2
            canvas = bg.copy()
            canvas.paste(fg, (fx, fy))
            out = str(assets_dir / f"{prefix}_portrait.jpg")
            canvas.save(out, quality=95)
            return out
        except Exception:
            return src

    def _burn_on_screen_text(src: str, text: str, *, prefix: str) -> str:
        """Burn on-screen text into the lower-third of a slide image.

        Returns the path to the modified image, or *src* unchanged on error.
        """
        if not text.strip():
            return src
        try:
            from PIL import Image, ImageDraw, ImageFont

            img = Image.open(src).convert("RGB")
            w, h = img.size
            draw = ImageDraw.Draw(img)

            # Pick a bold font.
            font_size = max(24, int(w * 0.038))
            font = None
            for fp in [
                r"C:\Windows\Fonts\seguisb.ttf",
                r"C:\Windows\Fonts\segoeuib.ttf",
                r"C:\Windows\Fonts\arialbd.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]:
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except Exception:
                    continue
            if font is None:
                font = ImageFont.load_default()

            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            margin_x = int(w * 0.06)
            max_text_w = w - 2 * margin_x

            # Word-wrap each line.
            wrapped: list[str] = []
            for line in lines:
                words = line.split()
                cur = ""
                for word in words:
                    test = f"{cur} {word}".strip()
                    bbox = draw.textbbox((0, 0), test, font=font)
                    if (bbox[2] - bbox[0]) > max_text_w and cur:
                        wrapped.append(cur)
                        cur = word
                    else:
                        cur = test
                if cur:
                    wrapped.append(cur)

            if not wrapped:
                return src

            line_h = int(font_size * 1.35)
            block_h = len(wrapped) * line_h
            pad_y = int(line_h * 0.5)
            pad_x = int(margin_x * 0.7)

            # Position: lower third of the frame (above caption zone).
            box_top = int(h * 0.38) - block_h // 2
            box_bottom = box_top + block_h + 2 * pad_y

            # Semi-transparent dark pill behind text.
            overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            # Find the widest line for pill width.
            max_line_w = 0
            for ln in wrapped:
                bbox = draw.textbbox((0, 0), ln, font=font)
                max_line_w = max(max_line_w, bbox[2] - bbox[0])
            pill_w = max_line_w + 2 * pad_x
            pill_x = (w - pill_w) // 2
            od.rounded_rectangle(
                [pill_x, box_top, pill_x + pill_w, box_bottom],
                radius=int(font_size * 0.4),
                fill=(0, 0, 0, 160),
            )
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(img)

            # Draw text lines centered.
            y = box_top + pad_y
            for ln in wrapped:
                bbox = draw.textbbox((0, 0), ln, font=font)
                lw = bbox[2] - bbox[0]
                x = (w - lw) // 2
                # White text with thin dark outline for readability.
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]:
                    draw.text((x + dx, y + dy), ln, font=font, fill=(0, 0, 0))
                draw.text((x, y), ln, font=font, fill=(255, 255, 255))
                y += line_h

            out = str(assets_dir / f"{prefix}_osd.jpg")
            img.save(out, quality=95)
            return out
        except Exception:
            import traceback; traceback.print_exc()
            return src

    # ── 3. Search + download: first unique valid image wins ──

    # Locate a prior scripted-short output for the same audio (image reuse).
    def _find_prior_scripted_dir() -> Path | None:
        if not reuse_images:
            return None
        cwd = Path.cwd()
        stem_lower = audio_path.stem.lower()
        best: Path | None = None
        best_mtime = 0.0
        for d in cwd.iterdir():
            if not d.is_dir() or not d.name.startswith("output_"):
                continue
            if stem_lower not in d.name.lower():
                continue
            if d.resolve() == Path(out_dir).resolve():
                continue  # skip the current output dir
            assets_d = d / "assets"
            if not assets_d.is_dir():
                continue
            if not list(assets_d.glob("s00*")):
                continue
            m = d.stat().st_mtime
            if m > best_mtime:
                best_mtime = m
                best = d
        return best

    _reuse_src: Path | None = _find_prior_scripted_dir()
    if _reuse_src:
        print(f"[scripted_short] Reusing assets from: {_reuse_src.name}")
    elif reuse_images:
        print("[scripted_short] No prior run found — downloading fresh images")

    slides: list[Slide] = []
    last_image: str | None = None

    for i, sec in enumerate(tqdm(sections, desc="Searching images")):
        image_path: str | None = None
        info: dict | None = None

        # ── Priority: use user-provided image if pre-assigned ──
        if i in _user_image_map:
            _upath = _user_image_map[i]
            if Path(_upath).exists():
                import shutil as _shutil
                _dst = str(assets_dir / f"s{i:02d}_user{Path(_upath).suffix}")
                _shutil.copy2(_upath, _dst)
                image_path = _dst
                print(f"  [s{i:02d}] Using user-provided image (skipping search)")

        # ── Priority 2: reuse portrait image from prior run ──
        if image_path is None and _reuse_src is not None:
            _reuse_assets_dir = _reuse_src / "assets"
            # Prefer *_portrait.jpg (processed), fall back to any s{i:02d}* image.
            _reuse_cands = sorted(_reuse_assets_dir.glob(f"s{i:02d}_portrait*.jpg"))
            if not _reuse_cands:
                _reuse_cands = sorted(
                    p for p in _reuse_assets_dir.glob(f"s{i:02d}*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                    and "_osd" not in p.name
                )
            if _reuse_cands:
                import shutil as _shutil_reuse
                _rsrc = _reuse_cands[-1]
                _rdst = assets_dir / _rsrc.name
                if not _rdst.exists():
                    _shutil_reuse.copy2(_rsrc, _rdst)
                image_path = str(_rdst)
                print(f"  [s{i:02d}] Reused from prior run: {_rsrc.name}")

        # Try primary query, then topic-only fallback.
        queries = [sec.search_query]
        if effective_topic and effective_topic.lower() not in sec.search_query.lower():
            queries.append(f"{effective_topic} {sec.label}")
        if effective_topic:
            queries.append(effective_topic)

        for q in queries:
            if image_path:
                break
            print(f"  [s{i:02d}] querying: {q}")
            try:
                candidates = _search(q)
            except Exception:
                candidates = []

            # Pre-filter obviously bad candidates (posters, covers, etc.)
            filtered = [c for c in candidates if not _is_bad_candidate(c)]
            if not filtered:
                filtered = candidates  # if ALL rejected, try them anyway

            if not filtered:
                continue

            # Use agent panel to pick the best candidate.
            best_idx = _agent_pick_image(
                filtered,
                section_label=sec.label,
                section_text=sec.text,
                query=q,
                topic=effective_topic or "",
            )

            # Try the agent's pick first, then fall through to others.
            _order = [best_idx] + [j for j in range(len(filtered)) if j != best_idx]
            for ci in _order:
                if ci >= len(filtered):
                    continue
                cand = filtered[ci]
                try:
                    p = _download_unique(cand, prefix=f"s{i:02d}")
                except Exception:
                    p = None
                if p:
                    image_path = p
                    info = cand
                    if cand.get("page_url"):
                        used_source_pages.append(str(cand["page_url"]))
                    _pick_label = "agent pick" if ci == best_idx else f"fallback #{ci}"
                    print(f"  [s{i:02d}] ✓ got image ({_pick_label})")
                    break

        if not image_path:
            if last_image:
                image_path = last_image
            else:
                continue  # skip (should be very rare)

        image_path = _preprocess_portrait(image_path, prefix=f"s{i:02d}")
        last_image = image_path

        # On-screen text is now rendered as audio-synced ASS overlays at
        # render time (see captions.generate_on_screen_text_ass), NOT
        # burned into slide images.

        # Timing placeholder — set below via Whisper word-timestamp matching.
        slides.append(Slide(
            start=0.0,
            end=0.0,
            image_path=image_path,
            query=sec.search_query,
        ))

    if not slides:
        raise RuntimeError("Could not download any images for the scripted sections.")

    # ── Sync slide timing to audio via Whisper word-level alignment ──
    # Load Whisper word timestamps first — needed for both slide timing and
    # caption/OSD generation later.
    try:
        from .transcribe import ensure_word_timestamps
        word_data = ensure_word_timestamps(audio_path, model_name="small")
    except Exception:
        word_data = []

    if word_data:
        # For each section, find where its text actually starts in the audio
        # by searching for the opening words in the Whisper word stream.
        import difflib

        whisper_lower = [str(w.get("word", "")).strip().lower().rstrip(".,!?…;:'\"") for w in word_data]

        def _find_section_start(script_text: str, search_from: int = 0) -> float | None:
            """Find the Whisper timestamp where *script_text* starts."""
            sw = script_text.split()
            if not sw:
                return None
            # Try to match the first 3-5 words as a fingerprint.
            needle = [w.lower().rstrip(".,!?…;:'\"") for w in sw[:5]]
            best_score = 0.0
            best_idx = -1
            for i in range(search_from, len(whisper_lower) - len(needle) + 1):
                chunk = whisper_lower[i:i + len(needle)]
                score = difflib.SequenceMatcher(None, needle, chunk).ratio()
                if score > best_score:
                    best_score = score
                    best_idx = i
                if score >= 0.8:
                    break  # good enough — take first match
            if best_idx >= 0 and best_score >= 0.5:
                return float(word_data[best_idx]["start"])
            return None

        section_starts: list[float] = []
        search_cursor = 0
        for idx, sec in enumerate(sections):
            # Include label + on_screen_text before narration body so we
            # match the moment the speaker says "Number one, Vampires as
            # colonizers" rather than the later explanation body.
            import re as _re_inner
            _label = sec.label or ""
            _osd = sec.on_screen_text or ""
            _osd_clean = _re_inner.sub(r'^\d+[)\.]\s*', '', _osd).strip()
            _match_text = f"{_label} {_osd_clean} {sec.text}".strip()
            t = _find_section_start(_match_text, search_from=search_cursor)
            if t is not None:
                section_starts.append(t)
                # Advance cursor past this hit so sections stay ordered.
                for ci in range(search_cursor, len(whisper_lower)):
                    if float(word_data[ci]["start"]) >= t:
                        search_cursor = ci + 1
                        break
            else:
                # Fallback: interpolate from neighbors.
                section_starts.append(-1.0)  # placeholder, resolved below

        # Resolve any -1 placeholders by linear interpolation.
        for idx in range(len(section_starts)):
            if section_starts[idx] < 0:
                prev_t = section_starts[idx - 1] if idx > 0 else 0.0
                next_t = audio_duration
                for j in range(idx + 1, len(section_starts)):
                    if section_starts[j] >= 0:
                        next_t = section_starts[j]
                        break
                section_starts[idx] = round((prev_t + next_t) / 2, 3)

        print("[scripted_short] Slide timing (Whisper-aligned):")
        for idx in range(len(slides)):
            t_start = section_starts[idx]
            t_end = section_starts[idx + 1] if idx + 1 < len(section_starts) else float(audio_duration)
            slides[idx] = Slide(
                start=round(t_start, 3),
                end=round(t_end, 3),
                image_path=slides[idx].image_path,
                query=slides[idx].query,
            )
            print(f"  s{idx:02d} {t_start:.2f}–{t_end:.2f}s")

    else:
        # No Whisper data — fall back to word-count proportional.
        section_word_counts = [len(s.text.split()) for s in sections]
        total_words = sum(section_word_counts) or 1
        cumulative = 0
        for idx in range(len(slides)):
            t_start = round(cumulative / total_words * audio_duration, 3)
            cumulative += section_word_counts[idx] if idx < len(section_word_counts) else 0
            t_end = round(cumulative / total_words * audio_duration, 3) if idx + 1 < len(slides) else float(audio_duration)
            slides[idx] = Slide(start=t_start, end=t_end,
                                image_path=slides[idx].image_path, query=slides[idx].query)
        print("[scripted_short] Slide timing (word-count proportional fallback)")

    # Always start from 0, end at full duration.
    slides[0] = Slide(start=0.0, end=slides[0].end,
                      image_path=slides[0].image_path, query=slides[0].query)
    slides[-1] = Slide(start=slides[-1].start, end=float(audio_duration),
                       image_path=slides[-1].image_path, query=slides[-1].query)

    # ── 3b. Video Short: fill video entirely with YouTube clips ─────────
    #
    # Strategy — SHUFFLED TRAILERS:
    # 1) Download 2-3 official HD trailers for the topic
    # 2) Split each into ~6-8s segments, skip first 10s and last 5s
    # 3) Shuffle segments independently per trailer, then interleave
    #    round-robin so consecutive sub-slides use different trailers
    # 4) Keep each clip ≤ 8 s for copyright safety
    #
    if use_clips:
        import math as _clip_math
        import random as _clip_rng
        from .clip_tools import (
            search_video_clips,
            download_clip,
            download_clip_section,
            trim_clip,
            get_video_duration,
        )

        clip_dir = ensure_dir(out_dir / "clips")

        # ── Copyright-safe bounds ──
        _MAX_CLIP_S = 8.0   # max seconds per individual clip
        _MIN_CLIP_S = 3.0   # don't bother with clips shorter than this

        # Use the short topic slug (e.g. "sinners") for cleaner queries.
        _clip_topic = _topic_slug or effective_topic or "unknown"

        # ── Topic keyword set for relevance filtering ─────────────────
        _topic_kw_set: set[str] = set()
        if _clip_topic and _clip_topic != "unknown":
            _stop = {"the", "a", "an", "of", "in", "to", "and", "or", "for", "is",
                     "movie", "film", "show", "series", "tv", "part", "season"}
            _topic_kw_set = {
                w.lower() for w in _clip_topic.split()
                if len(w) >= 3 and w.lower() not in _stop
            }

        # Words that indicate commentary / reaction / review channels.
        _REJECT_WORDS = {
            "reaction", "react", "review", "explained", "breakdown", "analysis",
            "commentary", "opinion", "discuss", "theory", "theories", "ranking",
            "tier list", "video essay", "response", "rant", "hot take",
            "why i", "why you", "what i think", "unpopular opinion",
        }

        def _is_official_clip(title: str) -> bool:
            """Return True only if the title looks like an official clip/trailer."""
            t = title.lower()
            if _topic_kw_set and not any(kw in t for kw in _topic_kw_set):
                return False
            for rw in _REJECT_WORDS:
                if rw in t:
                    return False
            return True

        # ── Download + cache raw clip files ───────────────────────────
        _raw_cache: dict[str, Path | None] = {}

        def _download_raw(url: str, prefix: str) -> Path | None:
            if url in _raw_cache:
                return _raw_cache[url]
            raw = download_clip_section(
                url, clip_dir / "raw",
                start=0.0, end=180.0,
                prefix=prefix,
            )
            if raw is None:
                raw = download_clip(
                    url, clip_dir / "raw",
                    prefix=prefix,
                    max_duration=300.0,
                )
            _raw_cache[url] = raw
            return raw

        # ── Download 2-3 official trailers ────────────────────────────
        _trailer_paths: list[tuple[Path, float]] = []
        _trailer_urls_seen: set[str] = set()

        # ── Reuse raw trailers from a prior run if available ──────────
        _reuse_clips_done = False
        if _reuse_src is not None:
            _prior_raw = _reuse_src / "clips" / "raw"
            if _prior_raw.is_dir():
                import shutil as _clip_reuse_shutil
                _raw_dst = clip_dir / "raw"
                _raw_dst.mkdir(parents=True, exist_ok=True)
                for _tf in sorted(_prior_raw.glob("trailer_*.mp4")):
                    _dst = _raw_dst / _tf.name
                    if not _dst.exists():
                        _clip_reuse_shutil.copy2(_tf, _dst)
                    try:
                        _dur = get_video_duration(_dst)
                    except Exception:
                        _dur = 0.0
                    if _dur >= 10.0:
                        _trailer_paths.append((_dst, _dur))
                        print(f"  [trailer] ✓ reused from prior run: {_tf.name} ({_dur:.0f}s)")
                if _trailer_paths:
                    _reuse_clips_done = True
                    print(f"[video_short] Reused {len(_trailer_paths)} trailers from: {_reuse_src.name}")

        if not _reuse_clips_done:
            print(f"\n[video_short] Searching for official trailers of '{_clip_topic}'…")

            for tq in [f"{_clip_topic} official trailer",
                        f"{_clip_topic} trailer HD",
                        f"{_clip_topic} trailer"]:
                if len(_trailer_paths) >= 3:
                    break
                try:
                    results = search_video_clips(
                        tq, max_results=10, preferred_max_duration=600.0,
                        sort_by_views=False,
                    )
                    results = [r for r in results if _is_official_clip(r.title)]
                    for chosen in results:
                        if len(_trailer_paths) >= 3:
                            break
                        if chosen.url in _trailer_urls_seen:
                            continue
                        _trailer_urls_seen.add(chosen.url)
                        raw = _download_raw(chosen.url, f"trailer_{len(_trailer_paths)}")
                        if raw and get_video_duration(raw) >= 10.0:
                            _trailer_paths.append((raw, get_video_duration(raw)))
                            print(f"  [trailer] ✓ {chosen.title[:60]} "
                                  f"({get_video_duration(raw):.0f}s)")
                except Exception as _tq_err:
                    print(f"  [trailer_search] Error: {_tq_err}")

        # Build shuffled segment pool — interleave trailers round-robin.
        # (path, start, end, trailer_index) — index tracks which trailer each seg is from.
        _per_trailer_segs: list[list[tuple[Path, float, float, int]]] = []
        for ti, (tpath, tdur) in enumerate(_trailer_paths):
            segs: list[tuple[Path, float, float, int]] = []
            usable_start = min(10.0, tdur * 0.15)
            usable_end = max(usable_start + _MIN_CLIP_S, tdur - 5.0)
            cursor = usable_start
            while cursor + _MIN_CLIP_S <= usable_end:
                seg_end = min(cursor + _MAX_CLIP_S, usable_end)
                segs.append((tpath, cursor, seg_end, ti))
                cursor = seg_end
            if not segs and tdur >= _MIN_CLIP_S:
                segs.append((tpath, 0.0, min(tdur, _MAX_CLIP_S), ti))
            _clip_rng.shuffle(segs)
            if segs:
                _per_trailer_segs.append(segs)

        # Interleave: pick one segment from each trailer in turn so
        # consecutive sub-slides come from different trailers.
        _segment_pool: list[tuple[Path, float, float, int]] = []
        if _per_trailer_segs:
            max_len = max(len(s) for s in _per_trailer_segs)
            for i in range(max_len):
                for tsegs in _per_trailer_segs:
                    if i < len(tsegs):
                        _segment_pool.append(tsegs[i])
        _seg_idx = 0

        print(f"[video_short] {len(_trailer_paths)} trailers → "
              f"{len(_segment_pool)} shuffled segments ready")

        # ── Reuse cache: avoid re-trimming the same trailer segment ──
        import shutil as _clip_shutil
        _trim_cache: dict[tuple[str, float, float, int, int], Path] = {}

        def _get_or_trim(src: Path, start: float, duration: float,
                         dest: Path) -> Path | None:
            """Return a trimmed clip, reusing a cached file when possible."""
            cache_key = (str(src), round(start, 2), round(duration, 2),
                         video_width, video_height)
            cached = _trim_cache.get(cache_key)
            if cached and cached.exists():
                _clip_shutil.copy2(cached, dest)
                return dest
            trim_clip(
                src, dest,
                start=start, duration=duration,
                width=video_width, height=video_height,
                mute=True,
            )
            _trim_cache[cache_key] = dest
            return dest

        # ── Build clip sub-slides per section ─────────────────────────
        new_slides: list[Slide] = []
        print(f"\n[video_short] Building clips for {len(slides)} sections…")

        for si, slide in enumerate(slides):
            section_dur = slide.end - slide.start
            if section_dur < _MIN_CLIP_S:
                new_slides.append(slide)
                continue

            n_subclips = max(1, _clip_math.ceil(section_dur / _MAX_CLIP_S))
            sub_dur = section_dur / n_subclips
            section_clips_ok = 0

            for sub_i in range(n_subclips):
                sub_start = slide.start + sub_i * sub_dur
                sub_end = sub_start + sub_dur
                if sub_i == n_subclips - 1:
                    sub_end = slide.end

                actual_sub_dur = sub_end - sub_start
                clip_tag = f"s{si:02d}c{sub_i}"
                got_clip = False

                if _segment_pool:
                    pool_idx = _seg_idx % len(_segment_pool)
                    # Reshuffle when we wrap around for fresh ordering.
                    if _seg_idx > 0 and pool_idx == 0:
                        _clip_rng.shuffle(_segment_pool)
                    tpath, seg_s, seg_e, t_idx = _segment_pool[pool_idx]
                    _seg_idx += 1

                    seg_dur = min(seg_e - seg_s, actual_sub_dur)
                    trimmed = clip_dir / f"prepared_{clip_tag}.mp4"
                    try:
                        _get_or_trim(tpath, seg_s, seg_dur, trimmed)
                        actual_dur = get_video_duration(trimmed)
                        if actual_dur <= 0:
                            actual_dur = seg_dur

                        new_slides.append(Slide(
                            start=sub_start, end=sub_end,
                            image_path=slide.image_path,
                            query=f"{_clip_topic} trailer",
                            video_clip_path=str(trimmed),
                            video_clip_start=0.0,
                            video_clip_end=min(actual_dur, actual_sub_dur),
                            video_clip_mute=True,
                        ))
                        got_clip = True
                        section_clips_ok += 1
                        print(f"  [{clip_tag}] ✓ trailer#{t_idx} "
                              f"@{seg_s:.1f}-{seg_s + seg_dur:.1f}s ({seg_dur:.1f}s)")
                    except Exception as _trim_err:
                        print(f"  [{clip_tag}] trim error: {_trim_err}")

                if not got_clip:
                    new_slides.append(Slide(
                        start=sub_start, end=sub_end,
                        image_path=slide.image_path, query=slide.query,
                    ))
                    print(f"  [{clip_tag}] ✗ no clip — image fallback")

            print(f"  [section {si:02d}] {section_clips_ok}/{n_subclips} sub-clips filled")

        slides = new_slides
        n_clips = sum(1 for s in slides if s.video_clip_path)
        print(f"[video_short] {n_clips}/{len(slides)} total sub-slides have video clips")

    # ── 4. Write outputs ──
    write_json(out_dir / "timeline.json", [asdict(s) for s in slides])

    # Save the parsed script for reference.
    write_json(out_dir / "script_sections.json", [
        {"start": s.start, "end": s.end, "label": s.label, "text": s.text,
         "on_screen_text": s.on_screen_text, "search_query": s.search_query}
        for s in sections
    ])

    try:
        write_json(out_dir / "run_meta.json", {
            "audio_name": audio_path.name,
            "audio_stem": audio_path.stem,
            "audio_size": int(audio_stat.st_size),
            "audio_mtime": float(audio_stat.st_mtime),
            "render_audio_path": None,
            "video_type": ("video_short" if use_clips else "scripted_short") if int(video_height) > int(video_width) else ("video_long" if use_clips else "scripted_long"),
            "video_width": int(video_width),
            "video_height": int(video_height),
            "topic": effective_topic,
            "topic_type": None,
            "image_provider": image_provider,
            "created_at": time.time(),
            "cwd": os.getcwd(),
        })
    except Exception:
        pass

    print(f"[scripted_short] timeline.json written with {len(slides)} slides")
    return out_dir


# =====================================================================
# Top-N List Short
# =====================================================================

def run_top_list(
    *,
    audio_path: str | Path,
    out_dir: str | Path,
    title: str,
    list_count: int,
    item_images: dict[int, str],
    item_titles: dict[int, str] | None = None,
    video_width: int = 1080,
    video_height: int = 1920,
    cta_text: str = "SUBSCRIBE 👇",
    cta_seconds: float = 3.0,
) -> Path:
    """Build a timeline for a **Top-N List** short.

    Parameters
    ----------
    title : str
        Title displayed as the first text-only slide (e.g. "5 Bad movies from 2025").
    list_count : int
        How many items in the list (e.g. 5, 10).
    item_images : dict[int, str]
        Mapping of item number → image file path.  E.g. ``{5: "/path/img5.png", 4: ...}``.
    cta_text : str
        Text for the subscribe/CTA end screen.
    cta_seconds : float
        Duration of the CTA end screen in seconds.

    Flow
    ----
    1. Transcribe audio (Whisper, word-level timestamps).
    2. Detect "number N" utterances to segment the timeline.
    3. Build slides:
       - Title card (black bg + title text) — from start until narrator says "number <count>"
       - For each number detected:
         * Number card (text-only, e.g. "5") — from "number N" until narrator starts
           talking about the item
         * Item image — from when narrator starts talking until next "number N-1"
       - CTA card ("SUBSCRIBE") — last ``cta_seconds`` of the video
    4. Write ``timeline.json`` and ``run_meta.json``.

    Returns *out_dir*.
    """
    import re as _re

    from .slide_cards import render_text_card, burn_title_on_image

    _item_titles = item_titles or {}

    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = ensure_dir(out_dir / "assets")

    audio_stat = audio_path.stat()
    audio_duration = get_audio_duration_seconds(audio_path)

    # ── Swipe transition helper ──
    SWIPE_DURATION = 0.35  # seconds

    def _create_swipe_clip(
        img_from: str, img_to: str, out_path: str,
        w: int, h: int, dur: float = SWIPE_DURATION, fps: int = 30,
    ) -> str:
        """Render a short slide-left xfade transition video between two images."""
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        # Each input loops for dur*2 so xfade has enough frames.
        # xfade offset=dur means transition starts at dur into the first stream.
        # Output duration = dur*2 + dur*2 - dur = 3*dur, but we trim to dur.
        scale_crop = (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h}:(in_w-out_w)/2:(in_h-out_h)/2,"
            f"format=yuv420p,fps={fps}"
        )
        fc = (
            f"[0:v]{scale_crop},trim=duration={dur:.3f},setpts=PTS-STARTPTS[a];"
            f"[1:v]{scale_crop},trim=duration={dur:.3f},setpts=PTS-STARTPTS[b];"
            f"[a][b]xfade=transition=slideleft:duration={dur:.3f}:offset=0[out]"
        )
        cmd = [
            ffmpeg, "-y",
            "-loop", "1", "-i", img_from,
            "-loop", "1", "-i", img_to,
            "-filter_complex", fc,
            "-map", "[out]",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-t", f"{dur:.3f}",
            "-an",
            out_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=30)
        return out_path

    # ── 1. Transcribe ──
    from .transcribe import ensure_word_timestamps

    words = ensure_word_timestamps(audio_path, model_name="small")
    if not words:
        raise RuntimeError("Whisper returned no word-level timestamps")

    # ── 2. Detect "number N" utterances ──
    import re as _re

    _WORD_TO_NUM = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20,
    }
    _ORDINAL_TO_NUM = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    }

    def _clean(w: str) -> str:
        return w.strip().lower().strip(".,!?:;\"'()[]#")

    # Print first 40 words for debugging.
    _dbg_words = [_clean(w["word"]) for w in words[:40]]
    print(f"[top_list] First 40 words: {_dbg_words}")

    number_markers: list[dict] = []  # {"number": int, "start": float, "end": float}

    for i, w in enumerate(words):
        word_clean = _clean(w["word"])

        # Pattern 1: "number" followed by a digit or number-word.
        if word_clean == "number" and i + 1 < len(words):
            next_word = _clean(words[i + 1]["word"])
            num = None
            if next_word.isdigit():
                num = int(next_word)
            elif next_word in _WORD_TO_NUM:
                num = _WORD_TO_NUM[next_word]
            if num is not None and 1 <= num <= list_count:
                number_markers.append({
                    "number": num,
                    "start": float(w["start"]),
                    "end": float(words[i + 1]["end"]),
                })
                continue

        # Pattern 2: Standalone "#5" or "number5" as single token.
        m = _re.match(r"(?:#|number)(\d+)$", word_clean)
        if m:
            num = int(m.group(1))
            if 1 <= num <= list_count:
                number_markers.append({
                    "number": num,
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                })
                continue

        # Pattern 3: Ordinals — "first", "second", "fifth", etc.
        if word_clean in _ORDINAL_TO_NUM:
            num = _ORDINAL_TO_NUM[word_clean]
            if 1 <= num <= list_count:
                number_markers.append({
                    "number": num,
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                })
                continue

        # Pattern 4: Standalone digit ("5", "4", …) used as a countdown marker.
        # The narrator says the number by itself, followed by a noticeable pause
        # before describing the item.  We require ≥0.5 s gap AFTER the digit to
        # filter out digits embedded in sentences ("here are 5 of the worst",
        # "Nezha 1", "Superman 2").
        if word_clean.isdigit():
            num = int(word_clean)
            if 1 <= num <= list_count:
                gap_after = (
                    float(words[i + 1]["start"]) - float(w["end"])
                    if i + 1 < len(words) else 999.0
                )
                if gap_after >= 0.5:
                    number_markers.append({
                        "number": num,
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                    })
                    continue

    # Deduplicate: keep only the *last* occurrence of each number.
    # In countdown scripts the narrator may mention a number in passing
    # ("Nezha 2 was bad") before reaching the actual countdown marker;
    # keeping the last hit gives us the real marker position.
    last_by_num: dict[int, dict] = {}
    for m in number_markers:
        last_by_num[m["number"]] = m  # overwrites earlier hits
    number_markers = list(last_by_num.values())

    # Sort by start time.
    number_markers.sort(key=lambda m: m["start"])

    print(f"[top_list] Detected {len(number_markers)} number markers: {[m['number'] for m in number_markers]}")

    # ── Fallback: evenly split the timeline if no markers found ──
    if not number_markers:
        print("[top_list] No markers detected — falling back to even timeline split")
        # Use last word end as total duration.
        total_dur = max(float(w["end"]) for w in words)
        # Reserve ~3 s at start for title card and cta_seconds at end.
        intro_pad = min(3.0, total_dur * 0.08)
        body_dur = total_dur - intro_pad - cta_seconds
        if body_dur < list_count * 1.0:
            body_dur = total_dur - intro_pad  # drop CTA reservation
        seg_dur = body_dur / list_count
        for idx in range(list_count):
            num = list_count - idx  # countdown: 5, 4, 3, 2, 1
            seg_start = intro_pad + idx * seg_dur
            number_markers.append({
                "number": num,
                "start": seg_start,
                "end": seg_start + min(1.5, seg_dur * 0.25),
            })
        print(f"[top_list] Fallback markers: {[m['number'] for m in number_markers]}")

    # ── 3. Find where narrator starts talking after each number ──
    # After "number 5" is spoken (end of that phrase), the next word is the
    # start of the description — that's when we switch to the item image.

    def _find_next_word_after(end_time: float) -> float | None:
        """Find the start time of the first word after *end_time*."""
        for w in words:
            if float(w["start"]) > end_time + 0.05:
                return float(w["start"])
        return None

    # ── 4. Build slides ──
    slides: list[Slide] = []

    # 4a. Title card — from 0 until the first "number N" is spoken.
    title_card_path = str(assets_dir / "title_card.png")
    render_text_card(
        out_path=title_card_path,
        text=title,
        width=video_width,
        height=video_height,
        font_size_ratio=0.10,
        text_color=(255, 255, 0),
    )

    title_end = float(number_markers[0]["start"])
    if title_end < 0.3:
        title_end = 0.3  # Ensure at least a brief flash
    slides.append(Slide(
        start=0.0,
        end=title_end,
        image_path=title_card_path,
        query="title",
        headline=title,
        motion="hold",
    ))

    # 4b. Number cards + item images.
    for mi, marker in enumerate(number_markers):
        num = marker["number"]
        num_start = float(marker["start"])
        num_end = float(marker["end"])

        # When does the narrator start describing the item? (first word after "number N")
        desc_start = _find_next_word_after(num_end)
        if desc_start is None:
            desc_start = num_end + 0.5  # fallback

        # When does this item end? At the next number marker or near end of audio.
        if mi + 1 < len(number_markers):
            item_end = float(number_markers[mi + 1]["start"])
        else:
            # Last item — ends at audio_duration minus CTA time.
            item_end = max(desc_start + 1.0, audio_duration - cta_seconds)

        # Number card (text-only, e.g. "5.").
        num_card_path = str(assets_dir / f"num_card_{num:02d}.png")
        render_text_card(
            out_path=num_card_path,
            text=f"{num}.",
            width=video_width,
            height=video_height,
            font_size_ratio=0.35,
        )

        slides.append(Slide(
            start=num_start,
            end=desc_start,
            image_path=num_card_path,
            query=f"number {num}",
            headline=str(num),
            motion="hold",
        ))

        # Item image — display the user-provided image.
        img_path = item_images.get(num)
        if img_path and Path(img_path).exists():
            # Copy to assets dir for portability.
            dst = assets_dir / f"item_{num:02d}{Path(img_path).suffix}"
            if not dst.exists():
                shutil.copy2(img_path, dst)
            item_img = str(dst)
        else:
            # Fallback: render a placeholder card.
            item_img = str(assets_dir / f"item_placeholder_{num:02d}.png")
            render_text_card(
                out_path=item_img,
                text=f"#{num}",
                width=video_width,
                height=video_height,
                font_size_ratio=0.25,
            )

        # Portrait pre-processing (blur-behind for landscape images).
        _is_portrait = int(video_height) > int(video_width) * 1.1
        if _is_portrait:
            try:
                from PIL import Image as _PILImage, ImageFilter, ImageEnhance

                im = _PILImage.open(item_img).convert("RGB")
                src_w, src_h = im.size
                src_ratio = src_w / src_h
                tgt_ratio = int(video_width) / max(1, int(video_height))

                if src_ratio / max(0.01, tgt_ratio) > 1.4:
                    tw, th = int(video_width), int(video_height)
                    bg_scale = max(tw / src_w, th / src_h)
                    bg_w, bg_h = int(src_w * bg_scale + 0.5), int(src_h * bg_scale + 0.5)
                    bg = im.resize((bg_w, bg_h), _PILImage.LANCZOS)
                    bx, by = (bg_w - tw) // 2, (bg_h - th) // 2
                    bg = bg.crop((bx, by, bx + tw, by + th))
                    bg = bg.filter(ImageFilter.GaussianBlur(radius=30))
                    bg = ImageEnhance.Brightness(bg).enhance(0.35)
                    fg_scale = min(tw / src_w, th / src_h)
                    fg_w, fg_h = int(src_w * fg_scale + 0.5), int(src_h * fg_scale + 0.5)
                    fg = im.resize((fg_w, fg_h), _PILImage.LANCZOS)
                    fx, fy = (tw - fg_w) // 2, (th - fg_h) // 2
                    canvas = bg.copy()
                    canvas.paste(fg, (fx, fy))
                    portrait_path = str(assets_dir / f"item_{num:02d}_portrait.jpg")
                    canvas.save(portrait_path, quality=95)
                    item_img = portrait_path
            except Exception:
                pass

        # Burn yellow title heading onto the item image if provided.
        _item_title = _item_titles.get(num, "").strip()
        if _item_title:
            titled_path = str(assets_dir / f"item_{num:02d}_titled.png")
            burn_title_on_image(
                image_path=item_img,
                out_path=titled_path,
                title=_item_title,
                width=video_width,
                height=video_height,
                text_color=(255, 255, 0),
                position="top",
            )
            item_img = titled_path

        # ── Swipe transition: number card slides left to reveal item image ──
        _swipe_ok = False
        if (desc_start - num_start) > SWIPE_DURATION + 0.1:
            swipe_path = str(assets_dir / f"swipe_{num:02d}.mp4")
            try:
                _create_swipe_clip(
                    num_card_path, item_img, swipe_path,
                    w=video_width, h=video_height, dur=SWIPE_DURATION,
                )
                if Path(swipe_path).exists() and Path(swipe_path).stat().st_size > 1000:
                    # Shorten the number card to end before the swipe.
                    swipe_start = desc_start - SWIPE_DURATION
                    slides[-1] = Slide(
                        start=num_start,
                        end=swipe_start,
                        image_path=num_card_path,
                        query=f"number {num}",
                        headline=str(num),
                        motion="hold",
                    )
                    # Insert swipe transition as a video clip slide.
                    slides.append(Slide(
                        start=swipe_start,
                        end=desc_start,
                        image_path=item_img,
                        query=f"swipe {num}",
                        video_clip_path=swipe_path,
                        video_clip_start=0.0,
                        video_clip_end=SWIPE_DURATION,
                        video_clip_mute=True,
                        motion="hold",
                    ))
                    _swipe_ok = True
                    print(f"[top_list] Swipe transition for #{num}: {swipe_start:.2f}s–{desc_start:.2f}s")
            except Exception as _sw_err:
                print(f"[top_list] Swipe generation failed for #{num}: {_sw_err}")

        slides.append(Slide(
            start=desc_start,
            end=item_end,
            image_path=item_img,
            query=f"item {num} image",
            headline=_item_title or None,
            motion="zoom_in",
        ))

    # 4c. CTA / Subscribe card.
    cta_start = max(0.0, audio_duration - cta_seconds)
    cta_card_path = str(assets_dir / "cta_card.png")
    render_text_card(
        out_path=cta_card_path,
        text=cta_text,
        width=video_width,
        height=video_height,
        font_size_ratio=0.12,
        text_color=(255, 255, 0),
    )

    # Adjust last item slide to end before CTA.
    if slides and slides[-1].end > cta_start:
        last = slides[-1]
        slides[-1] = Slide(
            start=last.start,
            end=cta_start,
            image_path=last.image_path,
            query=last.query,
            headline=last.headline,
            motion=last.motion,
        )

    slides.append(Slide(
        start=cta_start,
        end=audio_duration,
        image_path=cta_card_path,
        query="cta",
        headline=cta_text,
        motion="hold",
    ))

    # ── 5. Write timeline + meta ──
    write_json(out_dir / "timeline.json", [asdict(s) for s in slides])

    try:
        write_json(out_dir / "run_meta.json", {
            "audio_name": audio_path.name,
            "audio_stem": audio_path.stem,
            "audio_size": int(audio_stat.st_size),
            "audio_mtime": float(audio_stat.st_mtime),
            "render_audio_path": None,
            "video_type": "top_list",
            "video_width": int(video_width),
            "video_height": int(video_height),
            "topic": title,
            "topic_type": None,
            "list_count": int(list_count),
            "image_provider": "user",
            "created_at": time.time(),
            "cwd": os.getcwd(),
        })
    except Exception:
        pass

    print(f"[top_list] timeline.json written with {len(slides)} slides")
    return out_dir


def run_top_list_silent(
    *,
    out_dir: str | Path,
    title: str,
    list_count: int,
    item_images: dict[int, str],
    item_titles: dict[int, str] | None = None,
    video_width: int = 1080,
    video_height: int = 1920,
    image_seconds: float = 4.0,
    number_seconds: float = 1.5,
    title_seconds: float = 3.0,
    cta_text: str = "SUBSCRIBE 👇",
) -> Path:
    """Build a timeline for a **silent Top-N List** (no narration).

    Each number card is shown for *number_seconds*, each item image for
    *image_seconds*.  A silent WAV is generated at the total duration so the
    renderer has an audio track to work with (BGM can be layered on top).

    Returns *out_dir*.
    """
    from .slide_cards import render_text_card, burn_title_on_image

    _item_titles = item_titles or {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = ensure_dir(out_dir / "assets")

    SWIPE_DURATION = 0.35

    def _create_swipe_clip(
        img_from: str, img_to: str, out_path: str,
        w: int, h: int, dur: float = SWIPE_DURATION, fps: int = 30,
    ) -> str:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        scale_crop = (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h}:(in_w-out_w)/2:(in_h-out_h)/2,"
            f"format=yuv420p,fps={fps}"
        )
        fc = (
            f"[0:v]{scale_crop},trim=duration={dur:.3f},setpts=PTS-STARTPTS[a];"
            f"[1:v]{scale_crop},trim=duration={dur:.3f},setpts=PTS-STARTPTS[b];"
            f"[a][b]xfade=transition=slideleft:duration={dur:.3f}:offset=0[out]"
        )
        cmd = [
            ffmpeg, "-y",
            "-loop", "1", "-i", img_from,
            "-loop", "1", "-i", img_to,
            "-filter_complex", fc,
            "-map", "[out]",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-t", f"{dur:.3f}",
            "-an",
            out_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=30)
        return out_path

    # ── Build slides ──
    slides: list[Slide] = []
    t = 0.0  # running clock

    # 1. Title card
    title_card_path = str(assets_dir / "title_card.png")
    render_text_card(
        out_path=title_card_path,
        text=title,
        width=video_width,
        height=video_height,
        font_size_ratio=0.10,
        text_color=(255, 255, 0),
    )
    slides.append(Slide(start=t, end=t + title_seconds, image_path=title_card_path,
                        query="title", headline=title, motion="hold"))
    t += title_seconds

    # 2. Number cards + item images (countdown: list_count → 1)
    for num in range(list_count, 0, -1):
        # Number card
        num_card_path = str(assets_dir / f"num_card_{num:02d}.png")
        render_text_card(
            out_path=num_card_path,
            text=f"{num}.",
            width=video_width,
            height=video_height,
            font_size_ratio=0.35,
        )

        num_start = t
        num_end = t + number_seconds
        slides.append(Slide(start=num_start, end=num_end, image_path=num_card_path,
                            query=f"number {num}", headline=str(num), motion="hold"))
        t = num_end

        # Item image
        img_path = item_images.get(num)
        if img_path and Path(img_path).exists():
            dst = assets_dir / f"item_{num:02d}{Path(img_path).suffix}"
            if not dst.exists():
                shutil.copy2(img_path, dst)
            item_img = str(dst)
        else:
            item_img = str(assets_dir / f"item_placeholder_{num:02d}.png")
            render_text_card(out_path=item_img, text=f"#{num}",
                             width=video_width, height=video_height, font_size_ratio=0.25)

        # Portrait pre-processing
        if int(video_height) > int(video_width) * 1.1:
            try:
                from PIL import Image as _PILImage, ImageFilter, ImageEnhance
                im = _PILImage.open(item_img).convert("RGB")
                src_w, src_h = im.size
                src_ratio = src_w / src_h
                tgt_ratio = int(video_width) / max(1, int(video_height))
                if src_ratio / max(0.01, tgt_ratio) > 1.4:
                    tw, th = int(video_width), int(video_height)
                    bg_scale = max(tw / src_w, th / src_h)
                    bg_w, bg_h = int(src_w * bg_scale + 0.5), int(src_h * bg_scale + 0.5)
                    bg = im.resize((bg_w, bg_h), _PILImage.LANCZOS)
                    bx, by = (bg_w - tw) // 2, (bg_h - th) // 2
                    bg = bg.crop((bx, by, bx + tw, by + th))
                    bg = bg.filter(ImageFilter.GaussianBlur(radius=30))
                    bg = ImageEnhance.Brightness(bg).enhance(0.35)
                    fg_scale = min(tw / src_w, th / src_h)
                    fg_w, fg_h = int(src_w * fg_scale + 0.5), int(src_h * fg_scale + 0.5)
                    fg = im.resize((fg_w, fg_h), _PILImage.LANCZOS)
                    fx, fy = (tw - fg_w) // 2, (th - fg_h) // 2
                    canvas = bg.copy()
                    canvas.paste(fg, (fx, fy))
                    portrait_path = str(assets_dir / f"item_{num:02d}_portrait.jpg")
                    canvas.save(portrait_path, quality=95)
                    item_img = portrait_path
            except Exception:
                pass

        # Burn title heading
        _item_title = _item_titles.get(num, "").strip()
        if _item_title:
            titled_path = str(assets_dir / f"item_{num:02d}_titled.png")
            burn_title_on_image(
                image_path=item_img, out_path=titled_path, title=_item_title,
                width=video_width, height=video_height,
                text_color=(255, 255, 0), position="top",
            )
            item_img = titled_path

        # Burn subscribe text on the last item image (num == 1)
        if num == 1:
            sub_path = str(assets_dir / f"item_{num:02d}_subscribe.png")
            burn_title_on_image(
                image_path=item_img, out_path=sub_path, title=cta_text,
                width=video_width, height=video_height,
                text_color=(255, 255, 0), position="bottom",
            )
            item_img = sub_path

        # Swipe transition
        if number_seconds > SWIPE_DURATION + 0.1:
            swipe_path = str(assets_dir / f"swipe_{num:02d}.mp4")
            try:
                _create_swipe_clip(num_card_path, item_img, swipe_path,
                                   w=video_width, h=video_height, dur=SWIPE_DURATION)
                if Path(swipe_path).exists() and Path(swipe_path).stat().st_size > 1000:
                    swipe_start = num_end - SWIPE_DURATION
                    slides[-1] = Slide(start=num_start, end=swipe_start,
                                       image_path=num_card_path, query=f"number {num}",
                                       headline=str(num), motion="hold")
                    slides.append(Slide(
                        start=swipe_start, end=num_end, image_path=item_img,
                        query=f"swipe {num}",
                        video_clip_path=swipe_path, video_clip_start=0.0,
                        video_clip_end=SWIPE_DURATION, video_clip_mute=True,
                        motion="hold",
                    ))
            except Exception:
                pass

        img_start = t
        img_end = t + image_seconds
        slides.append(Slide(start=img_start, end=img_end, image_path=item_img,
                            query=f"item {num} image", headline=_item_title or None,
                            motion="zoom_in"))
        t = img_end

    total_duration = t

    # 4. Generate silent WAV so the renderer has an audio track.
    silent_audio = out_dir / "silence.wav"
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ffmpeg, "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
            "-t", f"{total_duration:.3f}",
            "-c:a", "pcm_s16le",
            str(silent_audio),
        ], capture_output=True, timeout=30)
    except Exception:
        pass

    # 5. Write timeline + meta
    write_json(out_dir / "timeline.json", [asdict(s) for s in slides])

    try:
        write_json(out_dir / "run_meta.json", {
            "audio_name": "silence.wav",
            "audio_stem": "silence",
            "audio_size": int(silent_audio.stat().st_size) if silent_audio.exists() else 0,
            "audio_mtime": 0.0,
            "render_audio_path": str(silent_audio),
            "video_type": "top_list_silent",
            "video_width": int(video_width),
            "video_height": int(video_height),
            "topic": title,
            "topic_type": None,
            "list_count": int(list_count),
            "image_provider": "user",
            "created_at": time.time(),
            "cwd": os.getcwd(),
            "total_duration": total_duration,
        })
    except Exception:
        pass

    print(f"[top_list_silent] timeline.json written with {len(slides)} slides, total {total_duration:.1f}s")
    return out_dir


def run_top_list_clips(
    *,
    out_dir: str | Path,
    title: str,
    list_count: int,
    item_titles: dict[int, str],
    clip_seconds: float = 7.0,
    number_seconds: float = 2.5,
    title_seconds: float = 4.0,
    video_width: int = 1080,
    video_height: int = 1920,
    cta_text: str = "SUBSCRIBE 👇",
    reuse_clips_from: str | Path | None = None,
    force_fresh: bool = False,
    tts_intro: bool = True,
    tts_clip_titles: bool = True,
) -> Path:
    """Build a **silent Top-N List** with auto-searched trailer clips.

    No audio input required.  Each number card is shown for
    *number_seconds* and each trailer clip for *clip_seconds*.
    A silent WAV is generated so the renderer has an audio track.

    Parameters
    ----------
    item_titles : dict[int, str]
        Mapping of item number → search query (movie/show name).
        E.g. ``{5: "Ne Zha 2", 4: "Bugonia", ...}``.
    clip_seconds : float
        Duration per trailer clip (default 7 s).
    number_seconds : float
        Duration per number card (default 2.5 s).
    force_fresh : bool
        If True, skip reusing clips from prior runs (re-download everything).
    """
    from .slide_cards import render_text_card, burn_title_on_image
    from .clip_tools import (
        search_video_clips,
        download_clip,
        download_clip_section,
        trim_clip,
        get_video_duration,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = ensure_dir(out_dir / "assets")
    clip_dir = ensure_dir(out_dir / "clips")

    # ── Swipe transition helper ──
    SWIPE_DURATION = 0.35

    def _create_swipe_clip(
        img_from: str, img_to: str, out_path: str,
        w: int, h: int, dur: float = SWIPE_DURATION, fps: int = 30,
    ) -> str:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        scale_crop = (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h}:(in_w-out_w)/2:(in_h-out_h)/2,"
            f"format=yuv420p,fps={fps}"
        )
        fc = (
            f"[0:v]{scale_crop},trim=duration={dur:.3f},setpts=PTS-STARTPTS[a];"
            f"[1:v]{scale_crop},trim=duration={dur:.3f},setpts=PTS-STARTPTS[b];"
            f"[a][b]xfade=transition=slideleft:duration={dur:.3f}:offset=0[out]"
        )
        cmd = [
            ffmpeg, "-y",
            "-loop", "1", "-i", img_from,
            "-loop", "1", "-i", img_to,
            "-filter_complex", fc,
            "-map", "[out]",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-t", f"{dur:.3f}",
            "-an",
            out_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=30)
        return out_path

    # ── 1. Download trailer clips for each item ──
    _REJECT_WORDS = {
        "reaction", "react", "review", "explained", "breakdown", "analysis",
        "commentary", "opinion", "discuss", "theory", "theories", "ranking",
        "tier list", "video essay", "response", "rant", "hot take",
    }

    def _is_official_clip(title_str: str, query_kws: set[str]) -> bool:
        t = title_str.lower()
        if query_kws and not any(kw in t for kw in query_kws):
            return False
        for rw in _REJECT_WORDS:
            if rw in t:
                return False
        return True

    all_raw_trailers: dict[int, list[tuple[Path, float]]] = {}  # number → list of (raw_path, duration)

    def _burn_title_on_clip(
        clip_in: Path, clip_out: Path, title_text: str,
        w: int, h: int, clip_dur: float = 6.0,
        position: str = "top",
    ) -> Path:
        """Burn a yellow title with dark band onto a video clip using ffmpeg drawtext.

        Title text slowly zooms in (~15 %) over the clip duration.
        position: 'top' or 'bottom'.
        """
        import imageio_ffmpeg
        _ff = imageio_ffmpeg.get_ffmpeg_exe()

        # Font: prefer bold system fonts, fall back to bundled Asap.
        _fonts = [
            r"C\\:/Windows/Fonts/segoeuib.ttf",
            r"C\\:/Windows/Fonts/arialbd.ttf",
            r"C\\:/Windows/Fonts/impact.ttf",
        ]
        fontfile = _fonts[0]
        for _fp in _fonts:
            _real = _fp.replace("C\\:", "C:").replace("/", "\\")
            if Path(_real).exists():
                fontfile = _fp
                break
        else:
            # Bundled font
            _bundled = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "Asap-Variable.ttf"
            if _bundled.exists():
                fontfile = str(_bundled).replace("\\", "/").replace(":", "\\:")

        font_size = max(32, int(w * 0.07))
        band_h = int(font_size * 2.5)
        if position == "bottom":
            band_y = int(h * 0.88)
        else:
            band_y = int(h * 0.06)
        # Escape special chars for ffmpeg drawtext
        safe_title = title_text.replace("'", "\u2019").replace(":", "\\:").replace("%", "%%")

        # Dark semi-transparent band (top) + static yellow text
        drawtext = (
            f"drawbox=y={band_y}:x=0:w={w}:h={band_h}:color=black@0.6:t=fill,"
            f"drawtext=fontfile='{fontfile}':"
            f"text='{safe_title}':"
            f"fontsize={font_size}:"
            f"fontcolor=yellow:"
            f"borderw=3:bordercolor=black:"
            f"x=(w-text_w)/2:"
            f"y={band_y}+(({band_h}-text_h)/2)"
        )

        cmd = [
            _ff, "-y", "-i", str(clip_in),
            "-vf", drawtext,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-an",
            str(clip_out),
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
            if clip_out.exists() and clip_out.stat().st_size > 1000:
                return clip_out
        except Exception as _e:
            print(f"[burn_title_clip] ffmpeg drawtext failed: {_e}")
        return clip_in  # fallback: untitled clip

    # ── Find prior output dir with same title to reuse clips ──
    _prior_clips_dir: Path | None = None
    if not force_fresh and reuse_clips_from is None:
        # Normalise the title into a lowercase slug for matching.
        import re as _re_slug
        _title_slug = _re_slug.sub(r'[^a-z0-9]+', '_', title.strip().lower()).strip('_')
        _cwd = Path.cwd()
        _best_prior: Path | None = None
        _best_mtime: float = 0.0
        for _d in _cwd.iterdir():
            if not _d.is_dir() or not _d.name.startswith("output_"):
                continue
            if _d.resolve() == Path(out_dir).resolve():
                continue
            _d_slug = _re_slug.sub(r'[^a-z0-9]+', '_', _d.name.lower()).strip('_')
            if _title_slug not in _d_slug:
                continue
            _prior_raw = _d / "clips" / "raw"
            if not _prior_raw.is_dir():
                continue
            if not list(_prior_raw.glob("item_*")):
                continue
            _m = _d.stat().st_mtime
            if _m > _best_mtime:
                _best_mtime = _m
                _best_prior = _d
        if _best_prior:
            _prior_clips_dir = _best_prior / "clips"
            print(f"[top_list_clips] Found prior clips in: {_best_prior.name}")

    # ── Reuse raw clips from prior run, explicit reuse dir, or own raw dir ──
    _reuse_raw_dir = Path(reuse_clips_from) / "raw" if reuse_clips_from else None
    _prior_raw_dir = _prior_clips_dir / "raw" if _prior_clips_dir else None
    _own_raw_dir = clip_dir / "raw"

    # Directories to check for existing raw clips (in priority order).
    _raw_check_dirs: list[Path | None] = [_own_raw_dir, _reuse_raw_dir, _prior_raw_dir]
    if force_fresh:
        _raw_check_dirs = []  # skip all reuse

    MAX_TRAILERS_PER_ITEM = 1  # one official HD trailer per item is enough

    for num in range(list_count, 0, -1):
        item_title = item_titles.get(num, "").strip()
        if not item_title:
            print(f"[top_list_clips] No title for #{num} — skipping clip search")
            continue

        # Collect up to MAX_TRAILERS_PER_ITEM raw trailer paths with their durations.
        raw_trailers: list[tuple[Path, float]] = []
        _seen_filenames: set[str] = set()

        # Try reusing raw clips already downloaded (collect up to MAX_TRAILERS_PER_ITEM).
        for _check_dir in _raw_check_dirs:
            if len(raw_trailers) >= MAX_TRAILERS_PER_ITEM:
                break
            if _check_dir and _check_dir.is_dir():
                for _rf in sorted(_check_dir.glob(f"item_{num:02d}_*")):
                    if _rf.name in _seen_filenames:
                        continue
                    _rd = get_video_duration(_rf)
                    if _rd >= 5.0:
                        _local_raw = clip_dir / "raw"
                        _local_raw.mkdir(parents=True, exist_ok=True)
                        _local_copy = _local_raw / _rf.name
                        if not _local_copy.exists() and _rf.resolve() != _local_copy.resolve():
                            import shutil as _sh_cp
                            _sh_cp.copy2(_rf, _local_copy)
                            _dest = _local_copy
                        else:
                            _dest = _rf
                        raw_trailers.append((_dest, _rd))
                        _seen_filenames.add(_rf.name)
                        print(f"  [clip #{num}] ✓ reused trailer {len(raw_trailers)} from {_check_dir.parent.name} ({_rd:.0f}s)")
                        if len(raw_trailers) >= MAX_TRAILERS_PER_ITEM:
                            break

        # If we still need more trailers, search + download.
        if len(raw_trailers) < MAX_TRAILERS_PER_ITEM:
            _stop = {"the", "a", "an", "of", "in", "to", "and", "or", "for", "is",
                     "movie", "film", "show", "series", "tv", "part", "season"}
            kw_set = {
                w.lower() for w in item_title.split()
                if len(w) >= 3 and w.lower() not in _stop
            }
            _seen_urls: set[str] = set()

            for tq in [
                f"{item_title} official trailer",
                f"{item_title} official trailer 2",
                f"{item_title} trailer HD",
                f"{item_title} trailer",
            ]:
                if len(raw_trailers) >= MAX_TRAILERS_PER_ITEM:
                    break
                try:
                    results = search_video_clips(
                        tq, max_results=10, preferred_max_duration=600.0,
                        sort_by_views=False,
                    )
                    results = [
                        r for r in results
                        if _is_official_clip(r.title, kw_set) and r.url not in _seen_urls
                    ]
                    for chosen in results:
                        if len(raw_trailers) >= MAX_TRAILERS_PER_ITEM:
                            break
                        tidx = len(raw_trailers) + 1
                        raw_path = download_clip_section(
                            chosen.url, clip_dir / "raw",
                            start=0.0, end=180.0,
                            prefix=f"item_{num:02d}_t{tidx}",
                        )
                        if raw_path is None:
                            raw_path = download_clip(
                                chosen.url, clip_dir / "raw",
                                prefix=f"item_{num:02d}_t{tidx}",
                                max_duration=300.0,
                            )
                        if raw_path and get_video_duration(raw_path) >= 5.0:
                            _seen_urls.add(chosen.url)
                            _rd = get_video_duration(raw_path)
                            raw_trailers.append((raw_path, _rd))
                            print(f"  [clip #{num}] ✓ trailer {tidx}: {chosen.title[:60]} ({_rd:.0f}s)")
                except Exception as _err:
                    print(f"  [clip #{num}] search error: {_err}")

        if not raw_trailers:
            print(f"  [clip #{num}] ✗ no trailers found for '{item_title}'")
            continue

        # Store raw trailers — clipping is done in batch below.
        all_raw_trailers[num] = raw_trailers

    print(f"[top_list_clips] Trailers downloaded for items: {sorted(all_raw_trailers.keys())}")

    # ── 2. Extract two 5s clips per item from its trailers (skip first 12s). ──
    import imageio_ffmpeg as _iff_mod
    _ff_exe = _iff_mod.get_ffmpeg_exe()
    _thumb_vf = (
        f"split=2[tbg][tfg];"
        f"[tbg]scale={video_width}:{video_height}:force_original_aspect_ratio=increase,"
        f"crop={video_width}:{video_height}:(in_w-out_w)/2:(in_h-out_h)/2,"
        f"gblur=sigma=20,setsar=1[tblurred];"
        f"[tfg]scale={video_width}:{video_height}:force_original_aspect_ratio=decrease,"
        f"setsar=1[tcontent];"
        f"[tblurred][tcontent]overlay=(W-w)/2:(H-h)/2,format=rgb24[tout]"
    )

    # item_clips[num] = clip_path
    item_clips: dict[int, Path] = {}

    for num in range(list_count, 0, -1):
        trailers = all_raw_trailers.get(num, [])
        if not trailers:
            print(f"  [clip #{num}] ✗ no trailers — skipping")
            continue

        found_clip: Path | None = None

        for tidx, (raw_path, raw_dur) in enumerate(trailers):
            seg = 12.0
            end_limit = raw_dur - 2.0

            if seg + clip_seconds <= end_limit:
                out_clip = clip_dir / f"item_{num:02d}_t{tidx+1}_c1.mp4"
                if not (out_clip.exists() and get_video_duration(out_clip) > 0):
                    try:
                        trim_clip(raw_path, out_clip, start=seg, duration=clip_seconds,
                                  width=video_width, height=video_height, mute=True)
                    except Exception as _te:
                        print(f"  [clip #{num}] trim error: {_te}")

                if out_clip.exists() and get_video_duration(out_clip) > 0:
                    found_clip = out_clip
                    print(f"  [clip #{num}] t{tidx+1}: {seg:.1f}s–{seg+clip_seconds:.1f}s")
                    break

        if found_clip:
            item_clips[num] = found_clip
        else:
            print(f"  [clip #{num}] ✗ could not extract a clip")

    # ── 3. Build slides: title card → (number card → clip) per item ──
    slides: list[Slide] = []
    t = 0.0

    def _make_thumb(clip_path: Path) -> str:
        _thumb = str(assets_dir / f"{clip_path.stem}_thumb.png")
        if not Path(_thumb).exists():
            try:
                subprocess.run([
                    _ff_exe, "-y", "-i", str(clip_path),
                    "-vframes", "1", "-filter_complex", _thumb_vf, "-map", "[tout]",
                    _thumb,
                ], capture_output=True, timeout=15)
            except Exception:
                pass
        return _thumb if Path(_thumb).exists() else str(clip_path)

    # Title card
    title_card_path = str(assets_dir / "title_card.png")
    render_text_card(
        out_path=title_card_path,
        text=title,
        width=video_width,
        height=video_height,
        font_size_ratio=0.10,
        text_color=(255, 255, 0),
    )
    slides.append(Slide(start=t, end=t + title_seconds, image_path=title_card_path,
                        query="title", headline=title, motion="hold"))
    t += title_seconds

    # Countdown: number card → clip (with title; subscribe on last)
    for num in range(list_count, 0, -1):
        clip_path = item_clips.get(num)
        if clip_path is None:
            continue
        _item_title = item_titles.get(num, "").strip()

        # Number card
        num_card_path = str(assets_dir / f"num_card_{num:02d}.png")
        render_text_card(
            out_path=num_card_path,
            text=f"{num}.",
            width=video_width,
            height=video_height,
            font_size_ratio=0.35,
        )
        slides.append(Slide(start=t, end=t + number_seconds, image_path=num_card_path,
                            query=f"number {num}", headline=str(num), motion="hold"))
        t += number_seconds

        # Burn item title onto clip
        if _item_title:
            titled_clip = clip_dir / f"item_{num:02d}_titled.mp4"
            clip_path = _burn_title_on_clip(
                clip_path, titled_clip, _item_title,
                w=video_width, h=video_height, clip_dur=clip_seconds,
            )

        # Burn subscribe CTA onto clip of last item (num == 1)
        if num == 1:
            sub_clip = clip_dir / f"item_{num:02d}_subscribe.mp4"
            clip_path = _burn_title_on_clip(
                clip_path, sub_clip, cta_text,
                w=video_width, h=video_height, clip_dur=clip_seconds,
                position="bottom",
            )

        # Clip slide
        dur_c = min(get_video_duration(clip_path), clip_seconds)
        slides.append(Slide(
            start=t, end=t + clip_seconds,
            image_path=_make_thumb(clip_path),
            query="trailer clip",
            headline=_item_title or None,
            video_clip_path=str(clip_path),
            video_clip_start=0.0,
            video_clip_end=dur_c,
            video_clip_mute=True,
            motion="hold",
        ))
        t += clip_seconds

    total_duration = t

    # 3. Generate silent WAV.
    silent_audio = out_dir / "silence.wav"
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ffmpeg, "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
            "-t", f"{total_duration:.3f}",
            "-c:a", "pcm_s16le",
            str(silent_audio),
        ], capture_output=True, timeout=30)
    except Exception:
        pass

    # 4. Write timeline + meta.
    write_json(out_dir / "timeline.json", [asdict(s) for s in slides])

    try:
        write_json(out_dir / "run_meta.json", {
            "audio_name": "silence.wav",
            "audio_stem": "silence",
            "audio_size": int(silent_audio.stat().st_size) if silent_audio.exists() else 0,
            "audio_mtime": 0.0,
            "render_audio_path": str(silent_audio),
            "video_type": "top_list_clips",
            "video_width": int(video_width),
            "video_height": int(video_height),
            "topic": title,
            "topic_type": None,
            "list_count": int(list_count),
            "image_provider": "youtube_clips",
            "created_at": time.time(),
            "cwd": os.getcwd(),
            "total_duration": total_duration,
        })
    except Exception:
        pass

    print(f"[top_list_clips] timeline.json written with {len(slides)} slides, total {total_duration:.1f}s")
    return out_dir
