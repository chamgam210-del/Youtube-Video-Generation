#!/usr/bin/env python3
"""Quick integration test for the video clip mixing feature."""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

# Ensure .env is loaded.
from dotenv import load_dotenv
load_dotenv()

from videogenerator.models import TranscriptSegment, VideoClipSuggestion
from videogenerator.clip_suggestions import suggest_video_clips
from videogenerator.clip_tools import (
    search_video_clips,
    _pick_best_clip_with_llm,
    download_clip,
    trim_clip,
    get_video_duration,
    prepare_clip_for_suggestion,
)


def test_step(name: str):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")


def main():
    # Use a known movie review topic for testing.
    topic = "The Housemaid 2025 movie"
    title = "Brutally Honest Review - The Housemaid"

    # Create fake transcript segments (simulating a ~60s review).
    segments = [
        TranscriptSegment(start=0.0, end=8.0, text="Welcome back everyone. Today we're talking about The Housemaid."),
        TranscriptSegment(start=8.0, end=18.0, text="This movie stars Sydney Sweeney and Amanda Seyfried in what's supposed to be a thriller."),
        TranscriptSegment(start=18.0, end=28.0, text="The plot follows a young woman who gets hired as a housemaid for a wealthy couple."),
        TranscriptSegment(start=28.0, end=38.0, text="Things start getting weird when she discovers the family's dark secrets."),
        TranscriptSegment(start=38.0, end=48.0, text="The cinematography is actually pretty good, I'll give them that."),
        TranscriptSegment(start=48.0, end=60.0, text="But overall, the script is weak and the ending is predictable. Not recommended."),
    ]

    out_dir = Path("test_clip_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: LLM Clip Suggestions ──
    test_step("LLM Clip Suggestions")
    try:
        suggestions = suggest_video_clips(
            segments=segments,
            topic=topic,
            title=title,
            video_type="review",
            audio_duration=60.0,
            max_clips=3,
            model="gpt-4o-mini",
        )
        print(f"Got {len(suggestions)} suggestions:")
        for i, s in enumerate(suggestions):
            print(f"  [{i}] {s.timeline_start:.1f}s-{s.timeline_end:.1f}s | query: {s.search_query!r} | mute: {s.mute} | reason: {s.reason[:80]}")
        
        # Save suggestions.
        with open(out_dir / "clip_suggestions.json", "w") as f:
            json.dump([{
                "timeline_start": s.timeline_start, "timeline_end": s.timeline_end,
                "search_query": s.search_query, "reason": s.reason, "mute": s.mute,
            } for s in suggestions], f, indent=2)
        print("  -> Saved to clip_suggestions.json")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        suggestions = []

    if not suggestions:
        print("\nNo suggestions generated. Cannot proceed with search/download test.")
        return

    # ── Step 2: Video Search ──
    test_step("Video Search (SerpAPI)")
    first_sug = suggestions[0]
    try:
        results = search_video_clips(
            first_sug.search_query,
            max_results=5,
            preferred_max_duration=120.0,
        )
        print(f"Got {len(results)} search results for: {first_sug.search_query!r}")
        for i, r in enumerate(results):
            print(f"  [{i}] {r.title[:60]} | {r.source} | dur={r.duration_seconds}s | {r.url[:60]}")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        results = []

    if not results:
        print("\nNo search results. Cannot proceed with download test.")
        return

    # ── Step 3: LLM Pick Best ──
    test_step("LLM Pick Best Clip")
    try:
        best_idx = _pick_best_clip_with_llm(
            results,
            search_query=first_sug.search_query,
            reason=first_sug.reason,
        )
        print(f"LLM picked index {best_idx}: {results[best_idx].title[:60]}")
        print(f"  URL: {results[best_idx].url}")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        best_idx = 0

    # ── Step 4: Download ──
    test_step("Download Clip (yt-dlp)")
    clip_dir = out_dir / "clips" / "raw"
    try:
        raw_path = download_clip(
            results[best_idx].url,
            clip_dir,
            prefix="test_clip",
            max_duration=120.0,
        )
        if raw_path:
            dur = get_video_duration(raw_path)
            size_mb = raw_path.stat().st_size / (1024 * 1024)
            print(f"  Downloaded: {raw_path.name} ({size_mb:.1f} MB, {dur:.1f}s)")
        else:
            print("  Download returned None (clip may have been filtered by duration).")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        raw_path = None

    if not raw_path:
        print("\nDownload failed. Cannot proceed with trim test.")
        return

    # ── Step 5: Trim ──
    test_step("Trim Clip (ffmpeg)")
    trimmed_path = out_dir / "clips" / "trimmed_test.mp4"
    target_dur = first_sug.timeline_end - first_sug.timeline_start
    try:
        trim_clip(
            raw_path,
            trimmed_path,
            start=5.0,  # skip first 5s (intro)
            duration=min(target_dur, 8.0),
            width=1920,
            height=1080,
            mute=True,
        )
        dur = get_video_duration(trimmed_path)
        size_mb = trimmed_path.stat().st_size / (1024 * 1024)
        print(f"  Trimmed: {trimmed_path.name} ({size_mb:.1f} MB, {dur:.1f}s)")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

    # ── Step 6: Full prepare_clip_for_suggestion ──
    test_step("Full Pipeline: prepare_clip_for_suggestion")
    if len(suggestions) > 1:
        test_sug = suggestions[1]  # Use a different suggestion to test full flow.
    else:
        test_sug = suggestions[0]
    try:
        pc = prepare_clip_for_suggestion(
            test_sug,
            dest_dir=out_dir / "clips" / "full_test",
            clip_index=0,
            width=1920,
            height=1080,
        )
        if pc:
            dur = get_video_duration(pc.path)
            print(f"  Prepared clip: {pc.path.name} ({dur:.1f}s)")
            print(f"  Timeline: {pc.timeline_start:.1f}s - {pc.timeline_end:.1f}s")
            print(f"  Source: {pc.source_title[:60]}")
            print(f"  URL: {pc.source_url}")
            print(f"  Muted: {pc.muted}")
        else:
            print("  prepare_clip_for_suggestion returned None (non-fatal).")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print("  TEST COMPLETE")
    print(f"{'='*60}")
    print(f"Output files in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
