from __future__ import annotations

from dataclasses import asdict

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
