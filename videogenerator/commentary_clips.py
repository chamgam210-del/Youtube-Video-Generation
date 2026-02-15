"""LLM-powered clip planning for **commentary** videos.

Commentary videos have two clip insertion modes:

* **reference** – the narrator explicitly references a specific clip
  (e.g. "here's the clip", "take a look at this", "let's watch").
  → find and insert that one specific clip.

* **compilation** – the narrator describes a broad reaction / news event
  (e.g. "everyone is losing their minds over this", "here's what people are saying").
  → grab several popular clips on the topic and concatenate them into a montage.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .models import CommentaryClipSuggestion, TranscriptSegment


# ── Shared LLM helper ───────────────────────────────────────────────────────

def _openai_chat(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.4,
    max_tokens: int = 3072,
    timeout_s: int = 90,
) -> str:
    import requests

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    # Strip markdown fences.
    if content.startswith("```"):
        first_nl = content.index("\n") if "\n" in content else 3
        content = content[first_nl + 1:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    return content


# ── Main suggestion function ────────────────────────────────────────────────

def suggest_commentary_clips(
    segments: list[TranscriptSegment],
    *,
    topic: str | None = None,
    title: str | None = None,
    audio_duration: float = 0.0,
    max_clips: int = 8,
    min_clip_seconds: float = 4.0,
    max_clip_seconds: float = 25.0,
    model: str = "gpt-4o-mini",
) -> list[CommentaryClipSuggestion]:
    """Analyze a commentary transcript and suggest clip insertions.

    Returns a list of :class:`CommentaryClipSuggestion` sorted by ``timeline_start``.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not segments:
        return []

    # Build compact transcript.
    lines = [f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in segments]
    transcript = "\n".join(lines)
    if len(transcript) > 14000:
        transcript = transcript[:14000] + "\n… (truncated)"

    system = (
        "You are a professional video editor planning clip insertions for a YouTube "
        "**commentary** video.\n\n"
        "The host records a voiceover reacting to / commenting on news, drama, or events. "
        "At certain moments the viewer should SEE the clip(s) being discussed.\n\n"
        "There are TWO types of clip insertion:\n\n"
        '1. **reference** – The host explicitly cues a specific clip:\n'
        '   - Trigger phrases (MUST detect ALL of these): "let\'s take a look at the clip", '
        '"here\'s the clip", "take a look", "watch this", "let\'s watch", '
        '"let\'s see", "check this out", "play the clip", "this is what she said", '
        '"this is what he said", "look at this", "roll the clip", etc.\n'
        '   - CRITICAL: Every time the host says one of these trigger phrases, you MUST '
        "create a reference clip insertion at that EXACT timestamp. Do NOT skip any.\n"
        '   - You must find the EXACT clip the host is referencing (the controversial statement, '
        "the interview moment, the news segment, etc.).\n"
        "   - Use the surrounding transcript context to determine WHAT clip the host is "
        "referring to, then craft a specific search query.\n"
        "   - Usually 1 clip, 4–15 seconds.\n\n"
        '2. **compilation** – The host describes a broad reaction or event:\n'
        '   - Trigger phrases: "everyone is losing their minds", "people are going crazy", '
        '"here\'s what people are saying", "reactions have been insane", '
        '"they went off", "they all lost their mind", '
        '"republicans/democrats are freaking out", "they just went off like one after the other", etc.\n'
        "   - You should grab MULTIPLE popular clips on that topic and compile them into a montage.\n"
        "   - Usually 3–5 clips, total 10–25 seconds.\n"
        "   - Provide the main search_query AND extra_queries (one per clip variant).\n\n"
        "CRITICAL RULES:\n"
        "- **SCAN EVERY LINE** of the transcript for trigger phrases. Do NOT miss any.\n"
        "- The clip insertion 'start' should be the EXACT timestamp where the host says "
        "the trigger phrase. The clip plays IMMEDIATELY after the cue.\n"
        f"- Suggest AT MOST {max_clips} insertions total.\n"
        f"- Each insertion should be {min_clip_seconds:.0f}–{max_clip_seconds:.0f} seconds.\n"
        "- Clips must NOT overlap.\n"
        "- For **reference** clips: search_query should be very specific "
        "(person name + what they said/did + context). Use the surrounding transcript "
        "to figure out what the host is talking about.\n"
        "- For **compilation** clips: search_query is the broad topic; "
        "extra_queries are specific source variations (different news outlets, reaction videos, etc.).\n"
        "- Set mute=false for news clips where keeping audio adds value (interviews, "
        "press conferences, news segments). Set mute=true for music/entertainment clips.\n"
        "- Space insertions out; don't cluster.\n"
        "- If the narrator never cues a clip or describes reactions, return an empty list [].\n\n"
        "Return ONLY valid JSON (no markdown). Schema:\n"
        "[\n"
        "  {\n"
        '    "start": <float>,\n'
        '    "end": <float>,\n'
        '    "search_query": "<string>",\n'
        '    "reason": "<string>",\n'
        '    "clip_type": "reference" | "compilation",\n'
        '    "num_clips": <int>,\n'
        '    "extra_queries": ["<string>", ...] | null,\n'
        '    "mute": <bool>\n'
        "  }\n"
        "]\n"
        "where start/end are seconds into the audio."
    )

    user_content = {
        "topic": (topic or "").strip(),
        "title": (title or "").strip(),
        "audio_duration": round(float(audio_duration), 1),
        "transcript": transcript,
    }

    content = _openai_chat(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
        ],
    )

    parsed = json.loads(content)
    if not isinstance(parsed, list):
        return []

    suggestions: list[CommentaryClipSuggestion] = []
    for item in parsed:
        try:
            start = float(item["start"])
            end = float(item["end"])
            query = str(item.get("search_query") or "").strip()
            reason = str(item.get("reason") or "").strip()
            clip_type = str(item.get("clip_type") or "reference").strip().lower()
            if clip_type not in {"reference", "compilation"}:
                clip_type = "reference"
            num_clips = int(item.get("num_clips") or (1 if clip_type == "reference" else 3))
            extra_queries = item.get("extra_queries")
            if extra_queries and isinstance(extra_queries, list):
                extra_queries = [str(q).strip() for q in extra_queries if str(q).strip()]
            else:
                extra_queries = None
            mute = bool(item.get("mute", True))

            dur = end - start
            if dur < min_clip_seconds * 0.5 or dur > max_clip_seconds * 2.0:
                continue
            dur = max(min_clip_seconds, min(max_clip_seconds, dur))
            end = start + dur

            if not query:
                continue

            # For compilations, enforce at least 2 clips.
            if clip_type == "compilation":
                num_clips = max(2, min(6, num_clips))
            else:
                num_clips = 1

            suggestions.append(
                CommentaryClipSuggestion(
                    timeline_start=start,
                    timeline_end=end,
                    search_query=query,
                    reason=reason,
                    clip_type=clip_type,
                    num_clips=num_clips,
                    extra_queries=extra_queries,
                    mute=mute,
                )
            )
        except (KeyError, ValueError, TypeError):
            continue

    suggestions.sort(key=lambda s: s.timeline_start)

    # Remove overlaps (keep first).
    cleaned: list[CommentaryClipSuggestion] = []
    for s in suggestions:
        if cleaned and s.timeline_start < cleaned[-1].timeline_end:
            continue
        cleaned.append(s)

    return cleaned[:max_clips]
