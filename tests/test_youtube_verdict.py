from videogenerator.models import TranscriptSegment
from videogenerator.youtube import _fallback_verdict_label


def test_fallback_verdict_label_neutral_returns_decent():
    segs = [
        TranscriptSegment(0.0, 1.0, "Here are my thoughts."),
        TranscriptSegment(1.0, 2.0, "It was fine overall."),
    ]
    assert _fallback_verdict_label(segments=segs) == "Decent"


def test_fallback_verdict_label_positive_returns_masterpiece():
    segs = [
        TranscriptSegment(0.0, 1.0, "This was amazing and fantastic."),
    ]
    assert _fallback_verdict_label(segments=segs) == "Masterpiece!"


def test_fallback_verdict_label_negative_returns_garbage():
    segs = [
        TranscriptSegment(0.0, 1.0, "This was terrible and boring."),
    ]
    assert _fallback_verdict_label(segments=segs) == "Garbage!"
