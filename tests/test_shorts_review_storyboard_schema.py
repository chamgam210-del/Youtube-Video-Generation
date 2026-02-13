import pytest

from videogenerator.llm_storyboard import (
    normalize_shorts_review_storyboard,
    parse_shorts_review_storyboard,
)


def test_shorts_review_schema_normalization_enforces_rules():
    raw = {
        "hook_frame": {"image_type": "poster", "text": "OSCAR BAIT?", "duration": 9, "motion": "none"},
        "beats": [
            {
                "timestamp": "0.7–2.9",
                "line": "But here's the thing: it's actually entertaining.",
                "image_type": "poster",  # intentionally same as hook
                "motion": "slow_zoom",
                "text": "Standards have changed a lot",  # too many words
                "priority": "tension",
                "importance": "high",
            },
            {
                "timestamp": "3.0-4.4",
                "line": "And that's honestly wild.",
                "image_type": "reaction_closeup",
                "motion": "slow_zoom_in",
                "text": "THIS IS WILD RIGHT NOW",  # too many words
                "priority": "payoff",
                "importance": "high",
            },
            {
                "timestamp": "4.4-5.8",
                "line": "Do you agree?",
                "image_type": "neutral",
                "motion": "slow_zoom",
                "text": "DO YOU AGREE",  # too many words
                "priority": "loop",
                "importance": "high",
            },
            {
                "timestamp": "5.8-7.2",
                "line": "extra beat",
                "image_type": "neutral",
                "motion": "slow_zoom",
                "text": "EXTRA WORDS HERE",  # too many words
                "priority": "verdict",
                "importance": "high",  # will be downgraded (cap=3)
            },
        ],
        "ending_frame": {"image_type": "neutral", "text": "AGREE? 👇", "duration": 9, "motion": "none"},
    }

    sb = parse_shorts_review_storyboard(raw)
    norm = normalize_shorts_review_storyboard(sb, hook_seconds=1.6, ending_seconds=1.2, max_beats=10)

    assert pytest.approx(norm.hook_frame.duration, abs=1e-6) == 1.6
    assert pytest.approx(norm.ending_frame.duration, abs=1e-6) == 1.2

    # Hook contrast: poster hook should force beat 1 into a close-up category.
    assert norm.hook_frame.image_type == "poster"
    assert norm.beats[0].image_type in {"closeup", "reaction_closeup"}

    # Loop-compat: ending background type follows hook.
    assert norm.ending_frame.image_type == norm.hook_frame.image_type

    # <=2 words per beat text.
    for b in norm.beats:
        if b.text.strip():
            assert len(b.text.split()) <= 2

    # Intent-based motion mapping.
    assert norm.beats[0].priority == "tension"
    assert norm.beats[0].motion == "snap_zoom"
    assert any(b.priority == "loop" and b.motion == "none" for b in norm.beats)

    # Cap highs to 3.
    assert sum(1 for b in norm.beats if b.importance == "high") <= 3
