from videogenerator.segments import merge_short_segments, select_evenly_spaced
from videogenerator.models import TranscriptSegment


def test_merge_short_segments_merges_until_min():
    segs = [
        TranscriptSegment(0.0, 1.0, "a"),
        TranscriptSegment(1.0, 2.0, "b"),
        TranscriptSegment(2.0, 6.5, "c"),
    ]
    merged = merge_short_segments(segs, min_seconds=3.0)
    assert len(merged) == 2
    assert merged[0].start == 0.0
    assert merged[0].end == 2.0


def test_select_evenly_spaced_limits_and_sorts():
    segs = [TranscriptSegment(i * 10.0, i * 10.0 + 5.0, str(i)) for i in range(10)]
    picked = select_evenly_spaced(segs, max_items=3, audio_duration=100.0)
    assert len(picked) == 3
    assert picked == sorted(picked, key=lambda s: s.start)
