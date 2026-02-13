from __future__ import annotations

from videogenerator.llm_storyboard import HighlightSpan, _clips_from_spans, _expand_span_to_sentence_boundaries
from videogenerator.models import TranscriptSegment


def _seg(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=float(start), end=float(end), text=text)


def test_expand_span_to_sentence_boundaries_expands_backwards_on_mid_thought_start() -> None:
    segments = [
        _seg(0.0, 3.0, "This movie is wild."),
        _seg(3.0, 6.0, "and the acting is even wilder."),
        _seg(6.0, 9.0, "Overall I had fun."),
    ]

    s, e = _expand_span_to_sentence_boundaries(segments, start_i=1, end_i=1)
    assert (s, e) == (0, 1)


def test_clips_from_spans_deduplicates_repeated_points() -> None:
    segments = [
        _seg(0.0, 4.0, "The cinematography is amazing."),
        _seg(4.0, 8.0, "It looks gorgeous."),
        _seg(8.0, 12.0, "The cinematography is amazing."),
        _seg(12.0, 16.0, "It looks gorgeous."),
        _seg(16.0, 20.0, "But the pacing drags."),
    ]

    spans = [
        HighlightSpan(start_i=0, end_i=1, reason="Visuals"),
        HighlightSpan(start_i=2, end_i=3, reason="Visuals again"),
        HighlightSpan(start_i=4, end_i=4, reason="Pacing"),
    ]

    clips = _clips_from_spans(
        segments,
        spans,
        audio_duration=20.0,
        max_total_seconds=60.0,
        min_clip_seconds=3.0,
        max_clip_seconds=20.0,
    )

    # The repeated visuals span should be dropped, leaving 2 clips.
    assert len(clips) == 2
    assert clips[0].start == 0.0
    assert clips[1].start >= 16.0


def test_clips_from_spans_respects_total_budget() -> None:
    segments = []
    t = 0.0
    for i in range(30):
        segments.append(_seg(t, t + 4.0, f"Sentence {i}."))
        t += 4.0

    spans = [HighlightSpan(start_i=i, end_i=i, reason=f"r{i}") for i in range(30)]

    clips = _clips_from_spans(
        segments,
        spans,
        audio_duration=t,
        max_total_seconds=25.0,
        min_clip_seconds=3.0,
        max_clip_seconds=20.0,
    )

    total = sum(c.end - c.start for c in clips)
    assert total <= 25.0 + 1e-6
