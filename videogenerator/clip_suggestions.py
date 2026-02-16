"""LLM-powered clip suggestion: analyze transcript and decide where video clips would enhance commentary."""

from __future__ import annotations

import json
import os
from typing import Any

from .models import TranscriptSegment, VideoClipSuggestion


def _openai_chat_completions(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout_s: int = 90,
    max_retries: int = 5,
) -> str:
    """Minimal OpenAI chat completions call (no SDK dependency)."""
    import requests
    import time as _time

    for _attempt in range(max_retries + 1):
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": 0.4, "max_tokens": 2048},
            timeout=int(timeout_s),
        )
        if resp.status_code == 429 and _attempt < max_retries:
            _wait = min(int(resp.headers.get("Retry-After", 0)) or (30 * (_attempt + 1)), 120)
            print(f"[LLM] 429 rate-limited, retrying in {_wait}s (attempt {_attempt + 1}/{max_retries})")
            _time.sleep(_wait)
            continue
        resp.raise_for_status()
        break
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    # Strip markdown code fences if present.
    c = content.strip()
    if c.startswith("```"):
        first_nl = c.index("\n") if "\n" in c else 3
        c = c[first_nl + 1 :]
        if c.endswith("```"):
            c = c[: -3]
        c = c.strip()
    return c


def suggest_video_clips(
    segments: list[TranscriptSegment],
    *,
    topic: str | None = None,
    title: str | None = None,
    video_type: str = "review",
    audio_duration: float = 0.0,
    max_clips: int = 6,
    min_clip_seconds: float = 3.0,
    max_clip_seconds: float = 12.0,
    model: str = "gpt-4o-mini",
) -> list[VideoClipSuggestion]:
    """Ask the LLM to suggest places in the commentary where a relevant video clip should be inserted.

    Returns a list of VideoClipSuggestion, sorted by timeline_start.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []

    if not segments:
        return []

    # Build a compact transcript representation.
    transcript_lines: list[str] = []
    for s in segments:
        transcript_lines.append(f"[{s.start:.1f}-{s.end:.1f}] {s.text}")
    transcript_text = "\n".join(transcript_lines)

    # Truncate for token budget.
    if len(transcript_text) > 12000:
        transcript_text = transcript_text[:12000] + "\n… (truncated)"

    system = (
        "You are a professional video editor planning B-roll video clip insertions for a YouTube "
        f"{'movie/show review' if video_type == 'review' else 'commentary'} video.\n\n"
        "The host's audio commentary plays throughout. At certain moments the viewer should see "
        "a relevant VIDEO CLIP (trailer footage, movie scene, behind-the-scenes, interview clip, etc.) "
        "instead of a static image.\n\n"
        "RULES:\n"
        f"- Suggest AT MOST {max_clips} clip insertions.\n"
        f"- Each clip should be {min_clip_seconds:.0f}–{max_clip_seconds:.0f} seconds long.\n"
        "- Clips should illustrate what the host is actively discussing.\n"
        "- Provide a concise, specific search query for finding the clip on YouTube/web "
        "(include the movie/show name + what scene/moment to show).\n"
        "- Clips should NOT overlap each other.\n"
        "- For movie/show reviews, focus on: key scenes discussed, trailer moments, "
        "action sequences, emotional beats, or behind-the-scenes footage.\n"
        "- Set mute=true for clips from copyrighted sources (trailers, movie scenes). "
        "Set mute=false only for clips that are clearly public domain or the host's own content.\n"
        "- Space clips out; don't cluster them.\n\n"
        "Return ONLY valid JSON (no markdown). Schema:\n"
        '[\n  {"start": <float>, "end": <float>, "search_query": "<string>", '
        '"reason": "<string>", "mute": <bool>}\n]\n'
        'where start/end are seconds into the commentary audio.'
    )

    user_content = {
        "topic": (topic or "").strip(),
        "title": (title or "").strip(),
        "audio_duration": round(float(audio_duration), 1),
        "transcript": transcript_text,
    }

    content = _openai_chat_completions(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
        ],
        timeout_s=90,
    )

    parsed = json.loads(content)
    if not isinstance(parsed, list):
        return []

    suggestions: list[VideoClipSuggestion] = []
    for item in parsed:
        try:
            start = float(item["start"])
            end = float(item["end"])
            query = str(item.get("search_query") or "").strip()
            reason = str(item.get("reason") or "").strip()
            mute = bool(item.get("mute", True))

            dur = end - start
            if dur < min_clip_seconds * 0.5 or dur > max_clip_seconds * 2.0:
                continue
            # Clamp.
            dur = max(min_clip_seconds, min(max_clip_seconds, dur))
            end = start + dur

            if not query:
                continue

            suggestions.append(
                VideoClipSuggestion(
                    timeline_start=float(start),
                    timeline_end=float(end),
                    search_query=query,
                    reason=reason,
                    mute=mute,
                )
            )
        except (KeyError, ValueError, TypeError):
            continue

    suggestions.sort(key=lambda s: s.timeline_start)

    # Remove overlaps (keep first).
    cleaned: list[VideoClipSuggestion] = []
    for s in suggestions:
        if cleaned and s.timeline_start < cleaned[-1].timeline_end:
            continue
        cleaned.append(s)

    return cleaned[:max_clips]
