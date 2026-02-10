from videogenerator.llm_storyboard import extract_review_highlights_with_llm
from videogenerator.models import TranscriptSegment


def test_extract_review_highlights_spoiler_warning_short_circuits_without_openai_key(monkeypatch):
    # Ensure no API key is present; spoiler path must not require it.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    segs = [
        TranscriptSegment(0.0, 5.0, "Intro setup"),
        TranscriptSegment(5.0, 9.0, "Before we start, spoiler warning: we will discuss the ending."),
        TranscriptSegment(9.0, 12.0, "Ok let's go."),
    ]

    clips = extract_review_highlights_with_llm(segs, audio_duration=120.0)
    assert len(clips) == 1
    assert clips[0].start == 0.0
    assert 39.0 <= clips[0].end <= 40.5


def test_extract_review_highlights_spoiler_warning_detects_slightly_after_one_minute(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    segs = [
        TranscriptSegment(0.0, 10.0, "Intro"),
        TranscriptSegment(60.24, 65.12, "Official spoiler alert now."),
        TranscriptSegment(65.12, 70.0, "Continuing"),
    ]

    clips = extract_review_highlights_with_llm(segs, audio_duration=180.0)
    assert len(clips) == 1
    assert clips[0].start == 0.0
    assert 39.0 <= clips[0].end <= 40.5
