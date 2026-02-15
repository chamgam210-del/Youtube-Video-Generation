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
    min_seg_seconds: float = 6.0,
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
    mix_video_clips: bool = False,
    max_video_clips: int = 6,
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
        try:
            if out_wav.exists() and out_wav.stat().st_size > 4096:
                # Heuristic cache: if it's newer than input, reuse.
                if out_wav.stat().st_mtime >= audio_in.stat().st_mtime:
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
    if vt not in {"review", "explainer", "shorts", "shorts_review", "commentary", "auto"}:
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
    if vt in {"explainer", "shorts", "shorts_review", "commentary", "auto"} and not effective_topic:
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
    if vt in {"explainer", "shorts", "shorts_review", "commentary"} and (topic_type is None or topic_type == "other"):
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

    def _hash_file(path_str: str) -> str:
        b = Path(path_str).read_bytes()
        return hashlib.sha256(b).hexdigest()

    def _search_candidates(qstr: str) -> list[dict]:
        if not qstr:
            return []
        if image_provider == "serpapi":
            if not serpapi_api_key:
                raise RuntimeError("--image-provider serpapi requires --serpapi-key or SERPAPI_API_KEY")
            from .serpapi_provider import search_commons_candidates_via_serpapi

            return search_commons_candidates_via_serpapi(
                qstr,
                api_key=serpapi_api_key,
                min_width=min_image_width,
                max_results=25,
            )
        if image_provider == "google_images":
            if not serpapi_api_key:
                raise RuntimeError("--image-provider google_images requires --serpapi-key or SERPAPI_API_KEY")
            from .serpapi_provider import search_google_images_candidates_via_serpapi

            return search_google_images_candidates_via_serpapi(
                qstr,
                api_key=serpapi_api_key,
                min_width=min_image_width,
                max_results=25,
            )
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
                score -= abs(math.log(max(1e-6, aspect / target_ratio)))

            title = str(c.get("title") or "")
            page = str(c.get("page_url") or "")
            blob = (title + " " + page).lower()

            # Penalize common low-value results.
            bad = ["poster", "logo", "wordmark", "cover", "album", "soundtrack", "dvd", "blu-ray", "bluray"]
            if any(b in blob for b in bad):
                score -= 2.5

            # Bonus for scene still / portrait hints.
            good = ["still", "scene", "screencap", "frame", "portrait", "close", "face"]
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
    if (reuse_source_dir is None) and seed_topic:
        try:
            seed_candidates = _search_candidates(seed_topic)
            seed_candidates = _rerank_candidates(seed_candidates, qstr=seed_topic)
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
        if vt in {"explainer", "shorts", "shorts_review"} and anchor:
            if q_llm:
                if anchor.lower() not in q_llm.lower():
                    q_llm = f"{anchor} {q_llm}".strip()
            else:
                q_llm = ""

        # If this looks like TV content, bias toward episode stills/cast.
        if vt in {"explainer", "shorts", "shorts_review"} and (topic_type == "tv_show"):
            if q_context and not any(k in q_context.lower() for k in ("still", "stills", "cast", "scene", "episode")):
                q_context = f"{q_context} TV series scene still".strip()
            if q_llm and not any(k in q_llm.lower() for k in ("still", "stills", "cast", "scene", "episode")):
                q_llm = f"{q_llm} TV series scene still".strip()

        # Shorts Review: bias toward in-scene imagery.
        if vt == "shorts_review":
            if q_context and not any(k in q_context.lower() for k in ("still", "stills", "scene", "screencap", "frame")):
                q_context = f"{q_context} scene still".strip()
            if q_context and not any(k in q_context.lower() for k in ("close", "portrait", "face")):
                q_context = f"{q_context} close up".strip()

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

    # ── Video clip mixing: LLM suggests clips → search → download → trim → assign ──
    if mix_video_clips and vt in {"review", "review_long"}:
        try:
            from .clip_suggestions import suggest_video_clips
            from .clip_tools import prepare_clip_for_suggestion

            clip_dir = ensure_dir(out_dir / "clips")

            suggestions = suggest_video_clips(
                segments=segments,
                topic=effective_topic or audio_path.stem,
                title=audio_path.stem,
                video_type=vt,
                audio_duration=timeline_duration,
                max_clips=max_video_clips,
                model=llm_model,
            )
            write_json(out_dir / "clip_suggestions.json", [
                {"timeline_start": s.timeline_start, "timeline_end": s.timeline_end,
                 "search_query": s.search_query, "reason": s.reason, "mute": s.mute}
                for s in suggestions
            ])

            prepared_clips = []
            for ci, sug in enumerate(suggestions):
                try:
                    pc = prepare_clip_for_suggestion(
                        sug,
                        dest_dir=clip_dir,
                        clip_index=ci,
                        width=video_width,
                        height=video_height,
                        llm_model=llm_model,
                    )
                    if pc is not None:
                        prepared_clips.append(pc)
                except Exception:
                    pass  # Non-fatal; skip this clip.

            # Assign prepared clips to matching slides.
            # For each prepared clip, find the slide whose time range best overlaps
            # and split it: [image before] [video clip] [image after].
            from .clip_tools import get_video_duration as _clip_dur

            for pc in prepared_clips:
                best_idx = -1
                best_overlap = 0.0
                for si, sl in enumerate(slides):
                    if sl.video_clip_path:
                        continue
                    overlap_start = max(sl.start, pc.timeline_start)
                    overlap_end = min(sl.end, pc.timeline_end)
                    overlap = max(0.0, overlap_end - overlap_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_idx = si

                if best_idx >= 0 and best_overlap > 0.5:
                    sl = slides[best_idx]
                    actual_clip_dur = _clip_dur(pc.path)
                    if actual_clip_dur <= 0:
                        actual_clip_dur = pc.timeline_end - pc.timeline_start

                    # Clamp clip duration to fit within the slide.
                    clip_dur = min(actual_clip_dur, sl.end - sl.start)

                    # Determine where the clip sits within the slide.
                    clip_start_in_tl = max(sl.start, pc.timeline_start)
                    clip_end_in_tl = min(sl.end, clip_start_in_tl + clip_dur)
                    clip_dur = clip_end_in_tl - clip_start_in_tl

                    new_slides: list[Slide] = []

                    # Image part before the clip.
                    if clip_start_in_tl - sl.start > 0.5:
                        new_slides.append(Slide(
                            start=sl.start, end=clip_start_in_tl,
                            image_path=sl.image_path, query=sl.query,
                            headline=sl.headline, subhead=sl.subhead,
                            source_page=sl.source_page, image_url=sl.image_url,
                            license_name=sl.license_name, license_url=sl.license_url,
                            attribution=sl.attribution, motion=sl.motion,
                            window_text=sl.window_text, window_keywords=sl.window_keywords,
                            queries_tried=sl.queries_tried,
                        ))

                    # Video clip slide.
                    new_slides.append(Slide(
                        start=clip_start_in_tl, end=clip_end_in_tl,
                        image_path=sl.image_path, query=sl.query,
                        headline=sl.headline, subhead=sl.subhead,
                        source_page=sl.source_page, image_url=sl.image_url,
                        license_name=sl.license_name, license_url=sl.license_url,
                        attribution=sl.attribution, motion=sl.motion,
                        window_text=sl.window_text, window_keywords=sl.window_keywords,
                        queries_tried=sl.queries_tried,
                        video_clip_path=str(pc.path),
                        video_clip_start=0.0,
                        video_clip_end=clip_dur,
                        video_clip_mute=pc.muted,
                    ))

                    # Image part after the clip.
                    if sl.end - clip_end_in_tl > 0.5:
                        new_slides.append(Slide(
                            start=clip_end_in_tl, end=sl.end,
                            image_path=sl.image_path, query=sl.query,
                            headline=sl.headline, subhead=sl.subhead,
                            source_page=sl.source_page, image_url=sl.image_url,
                            license_name=sl.license_name, license_url=sl.license_url,
                            attribution=sl.attribution, motion=sl.motion,
                            window_text=sl.window_text, window_keywords=sl.window_keywords,
                            queries_tried=sl.queries_tried,
                        ))

                    # Replace the original slide with the split parts.
                    slides[best_idx:best_idx + 1] = new_slides

            write_json(out_dir / "prepared_clips.json", [
                {"path": str(pc.path), "timeline_start": pc.timeline_start,
                 "timeline_end": pc.timeline_end, "source_url": pc.source_url,
                 "source_title": pc.source_title, "search_query": pc.search_query,
                 "muted": pc.muted}
                for pc in prepared_clips
            ])
        except Exception as exc:
            # Video clip mixing is best-effort; don't fail the whole pipeline.
            import traceback
            traceback.print_exc()

    # ── Commentary clip insertion: always-on for commentary type ──────────────
    if vt == "commentary":
        # Commentary always uses the commentary-specific LLM to find reference clips
        # and build compilation montages.
        try:
            from .commentary_clips import suggest_commentary_clips
            from .clip_tools import (
                prepare_clip_for_suggestion as _prep_ref,
                prepare_compilation_clip as _prep_comp,
                get_video_duration as _clip_dur,
                PreparedClip,
            )
            from .models import VideoClipSuggestion as _VCS

            clip_dir = ensure_dir(out_dir / "clips")

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

            prepared_clips: list[PreparedClip] = []
            for ci, sug in enumerate(csug):
                try:
                    if sug.clip_type == "compilation":
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
                            sort_by_views=True,
                            mode="commentary",
                        )
                    if pc is not None:
                        prepared_clips.append(pc)
                except Exception:
                    pass

            # Assign prepared clips to matching slides (same logic as review mixing).
            for pc in prepared_clips:
                best_idx = -1
                best_overlap = 0.0
                for si, sl in enumerate(slides):
                    if sl.video_clip_path:
                        continue
                    overlap_start = max(sl.start, pc.timeline_start)
                    overlap_end = min(sl.end, pc.timeline_end)
                    overlap = max(0.0, overlap_end - overlap_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_idx = si

                if best_idx >= 0 and best_overlap > 0.5:
                    sl = slides[best_idx]
                    actual_clip_dur = _clip_dur(pc.path)
                    if actual_clip_dur <= 0:
                        actual_clip_dur = pc.timeline_end - pc.timeline_start
                    clip_dur = min(actual_clip_dur, sl.end - sl.start)
                    clip_start_in_tl = max(sl.start, pc.timeline_start)
                    clip_end_in_tl = min(sl.end, clip_start_in_tl + clip_dur)
                    clip_dur = clip_end_in_tl - clip_start_in_tl

                    new_slides: list[Slide] = []
                    if clip_start_in_tl - sl.start > 0.5:
                        new_slides.append(Slide(
                            start=sl.start, end=clip_start_in_tl,
                            image_path=sl.image_path, query=sl.query,
                            headline=sl.headline, subhead=sl.subhead,
                            source_page=sl.source_page, image_url=sl.image_url,
                            license_name=sl.license_name, license_url=sl.license_url,
                            attribution=sl.attribution, motion=sl.motion,
                            window_text=sl.window_text, window_keywords=sl.window_keywords,
                            queries_tried=sl.queries_tried,
                        ))
                    new_slides.append(Slide(
                        start=clip_start_in_tl, end=clip_end_in_tl,
                        image_path=sl.image_path, query=sl.query,
                        headline=sl.headline, subhead=sl.subhead,
                        source_page=sl.source_page, image_url=sl.image_url,
                        license_name=sl.license_name, license_url=sl.license_url,
                        attribution=sl.attribution, motion=sl.motion,
                        window_text=sl.window_text, window_keywords=sl.window_keywords,
                        queries_tried=sl.queries_tried,
                        video_clip_path=str(pc.path),
                        video_clip_start=0.0,
                        video_clip_end=clip_dur,
                        video_clip_mute=pc.muted,
                    ))
                    if sl.end - clip_end_in_tl > 0.5:
                        new_slides.append(Slide(
                            start=clip_end_in_tl, end=sl.end,
                            image_path=sl.image_path, query=sl.query,
                            headline=sl.headline, subhead=sl.subhead,
                            source_page=sl.source_page, image_url=sl.image_url,
                            license_name=sl.license_name, license_url=sl.license_url,
                            attribution=sl.attribution, motion=sl.motion,
                            window_text=sl.window_text, window_keywords=sl.window_keywords,
                            queries_tried=sl.queries_tried,
                        ))
                    slides[best_idx:best_idx + 1] = new_slides

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

    # If we inserted pivot pauses, generate a padded audio file for rendering.
    if vt == "shorts_review" and audio_pause_insertions:
        try:
            padded = _ensure_audio_with_pauses(audio_in=audio_path, insertions=audio_pause_insertions)
            if padded is not None and padded.exists() and padded.stat().st_size > 4096:
                render_audio_override = padded
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
        slides = [
            Slide(
                start=0.0,
                end=float(slides[0].start),
                image_path=slides[0].image_path,
                query=slides[0].query,
                headline=slides[0].headline,
                subhead=slides[0].subhead,
                source_page=slides[0].source_page,
                image_url=slides[0].image_url,
                license_name=slides[0].license_name,
                license_url=slides[0].license_url,
                attribution=slides[0].attribution,
                motion=slides[0].motion,
                window_text=slides[0].window_text,
                window_keywords=slides[0].window_keywords,
                queries_tried=slides[0].queries_tried,
                video_clip_path=slides[0].video_clip_path,
                video_clip_start=slides[0].video_clip_start,
                video_clip_end=slides[0].video_clip_end,
                video_clip_mute=slides[0].video_clip_mute,
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
