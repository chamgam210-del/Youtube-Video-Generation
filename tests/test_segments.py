from videogenerator.segments import bucketize_segments, merge_short_segments, select_evenly_spaced
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


def test_bucketize_segments_targets_fast_beats():
    # 40s of ~0.5s segments => should bucket into ~1-2s beats.
    segs = [TranscriptSegment(i * 0.5, (i + 1) * 0.5, f"w{i}") for i in range(80)]
    buckets = bucketize_segments(segs, target_seconds=1.7, min_seconds=1.2, max_seconds=2.0)
    assert buckets
    for b in buckets:
        dur = b.end - b.start
        assert dur >= 1.0  # allow slight undershoot depending on boundaries
        assert dur <= 2.5  # allow slight overshoot with edge conditions


def test_bucketize_segments_splits_very_long_segment():
    segs = [TranscriptSegment(0.0, 10.0, "lots of words here " * 10)]
    buckets = bucketize_segments(segs, target_seconds=1.7, min_seconds=1.2, max_seconds=2.0)
    assert len(buckets) >= 4
    assert buckets[0].start == 0.0
    assert buckets[-1].end == 10.0
