from __future__ import annotations

from dataclasses import asdict
import math

from .models import TranscriptSegment


def merge_short_segments(segments: list[TranscriptSegment], min_seconds: float) -> list[TranscriptSegment]:
    if not segments:
        return []

    merged: list[TranscriptSegment] = []
    cur_start = segments[0].start
    cur_end = segments[0].end
    cur_text = segments[0].text

    for seg in segments[1:]:
        cur_len = cur_end - cur_start
        seg_len = seg.end - seg.start

        # If current is already long enough, flush it.
        if cur_len >= min_seconds:
            merged.append(TranscriptSegment(start=cur_start, end=cur_end, text=cur_text.strip()))
            cur_start, cur_end, cur_text = seg.start, seg.end, seg.text
            continue

        # If the next segment is already long enough, don't absorb it into the short one.
        if seg_len >= min_seconds:
            merged.append(TranscriptSegment(start=cur_start, end=cur_end, text=cur_text.strip()))
            cur_start, cur_end, cur_text = seg.start, seg.end, seg.text
            continue

        # Otherwise, merge to build up duration.
        cur_end = max(cur_end, seg.end)
        cur_text = (cur_text + " " + seg.text).strip()

    merged.append(TranscriptSegment(start=cur_start, end=cur_end, text=cur_text.strip()))
    return merged


def select_evenly_spaced(
    segments: list[TranscriptSegment],
    max_items: int,
    audio_duration: float,
) -> list[TranscriptSegment]:
    if not segments:
        return []
    if max_items <= 0:
        return []
    if len(segments) <= max_items:
        return segments

    total = max(audio_duration, segments[-1].end)
    if total <= 0:
        return segments[:max_items]

    # Select by target times across the timeline
    targets = [(i + 0.5) * total / max_items for i in range(max_items)]
    out: list[TranscriptSegment] = []
    used: set[int] = set()

    for t in targets:
        best_idx = 0
        best_dist = float("inf")
        for i, seg in enumerate(segments):
            if i in used:
                continue
            mid = 0.5 * (seg.start + seg.end)
            dist = abs(mid - t)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        used.add(best_idx)
        out.append(segments[best_idx])

    out.sort(key=lambda s: s.start)
    return out


def segments_to_json(segments: list[TranscriptSegment]) -> list[dict]:
    return [asdict(s) for s in segments]


def split_long_segment(seg: TranscriptSegment, *, max_seconds: float) -> list[TranscriptSegment]:
    """Split a single long segment into multiple time-contiguous segments.

    Whisper occasionally emits long segments; for fast-cut Shorts we want a steady beat.
    We keep boundaries time-based and split text by word chunks (best-effort).
    """

    dur = float(seg.end - seg.start)
    if dur <= 0:
        return [seg]

    max_s = max(0.2, float(max_seconds))
    if dur <= max_s:
        return [seg]

    n = max(2, int(math.ceil(dur / max_s)))
    words = str(seg.text or "").strip().split()
    if not words:
        words = [""]

    out: list[TranscriptSegment] = []
    for i in range(n):
        a = seg.start + (dur * i / n)
        b = seg.start + (dur * (i + 1) / n)
        w0 = int(round(len(words) * i / n))
        w1 = int(round(len(words) * (i + 1) / n))
        chunk = " ".join(words[w0:w1]).strip() if w1 > w0 else " ".join(words[w0:w0 + 6]).strip()
        out.append(TranscriptSegment(start=float(a), end=float(b), text=chunk))
    return out


def bucketize_segments(
    segments: list[TranscriptSegment],
    *,
    target_seconds: float,
    min_seconds: float,
    max_seconds: float,
) -> list[TranscriptSegment]:
    """Merge/split segments into near-uniform buckets.

    Intended for Shorts-style pacing: each bucket becomes a slide beat.
    """

    if not segments:
        return []

    tgt = max(0.2, float(target_seconds))
    mn = max(0.2, min(float(min_seconds), tgt))
    mx = max(mn, float(max_seconds))

    # First, split any overly-long segments.
    flat: list[TranscriptSegment] = []
    for s in segments:
        flat.extend(split_long_segment(s, max_seconds=mx))

    out: list[TranscriptSegment] = []
    cur_start = float(flat[0].start)
    cur_end = float(flat[0].end)
    cur_text = str(flat[0].text or "")

    for seg in flat[1:]:
        cur_len = float(cur_end - cur_start)
        next_len = float(seg.end - cur_start)

        # Flush once we hit the target.
        if cur_len >= tgt:
            out.append(TranscriptSegment(start=cur_start, end=cur_end, text=cur_text.strip()))
            cur_start, cur_end, cur_text = float(seg.start), float(seg.end), str(seg.text or "")
            continue

        # If adding the next segment would exceed max, flush as long as we're not too short.
        if next_len > mx and cur_len >= mn:
            out.append(TranscriptSegment(start=cur_start, end=cur_end, text=cur_text.strip()))
            cur_start, cur_end, cur_text = float(seg.start), float(seg.end), str(seg.text or "")
            continue

        # Otherwise, merge.
        cur_end = float(max(cur_end, seg.end))
        cur_text = (cur_text + " " + str(seg.text or "")).strip()

    out.append(TranscriptSegment(start=cur_start, end=cur_end, text=cur_text.strip()))
    return out
