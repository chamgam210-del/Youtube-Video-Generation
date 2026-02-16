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
    min_clip_seconds: float = 10.0,
    max_clip_seconds: float = 25.0,
    model: str = "gpt-4o-mini",
) -> list[CommentaryClipSuggestion]:
    """Analyze a commentary transcript and suggest clip insertions.

    Returns a list of :class:`CommentaryClipSuggestion` sorted by ``timeline_start``.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not segments:
        return []

    # Use gpt-4o for commentary analysis — transcript understanding is critical.
    # Override only if the caller passed the default mini model.
    commentary_model = "gpt-4o" if model == "gpt-4o-mini" else model

    # Build compact transcript.
    lines = [f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in segments]
    transcript = "\n".join(lines)
    if len(transcript) > 14000:
        transcript = transcript[:14000] + "\n… (truncated)"

    system = (
        "You are a professional video editor planning clip insertions for a YouTube "
        "**commentary** video.\n\n"
        "## YOUR JOB\n"
        "1. **Read and DEEPLY UNDERSTAND the transcript.** Figure out WHO and WHAT the "
        "host is talking about — the people, events, controversies, reactions.\n"
        "2. **ONLY insert clips where the host EXPLICITLY CUES one.** If the host "
        "never cues a clip, return an EMPTY list [].\n"
        "3. **Generate highly specific YouTube search queries** that will find the "
        "EXACT clips viewers would want to see.\n\n"
        "## CRITICAL RULE: EXPLICIT CUE PHRASES ONLY\n"
        "You MUST ONLY insert a clip at a timestamp where the host says one of these "
        "CUE PHRASES (or something very similar):\n"
        '  "let\'s look at", "let\'s take a look", "let\'s watch", "here\'s the clip", '
        '"watch this", "check this out", "play the clip", "roll the clip", '
        '"look at this", "this is what he/she said", "let me show you", '
        '"take a look at this", "let\'s see", "here\'s what happened"\n\n'
        "**If the host does NOT use a cue phrase, DO NOT insert a clip there.**\n"
        "Do NOT insert clips just because the host mentions a person or event.\n"
        "Do NOT insert clips at random points for visual interest.\n"
        "The host must INVITE the viewer to watch something.\n\n"
        "## TWO TYPES OF CLIP INSERTION\n\n"
        '### 1. **reference** – The host cues a SPECIFIC clip\n'
        '   The host says something like "let\'s look at what he said" or '
        '"here\'s the clip" about ONE specific person/moment.\n'
        "   → Find THE specific clip being referenced.\n"
        "   → search_query must be VERY specific: include the person's full name + "
        "what they said/did + the show/event name.\n"
        "   → num_clips: **1**\n"
        "   → Duration: **8–15 seconds**\n\n"
        '### 2. **compilation** – The host cues clips from a GROUP of people\n'
        '   The host says something like "let\'s look at some of these republican '
        'reactions" or "let\'s see what people are saying" about MULTIPLE people.\n'
        "   → You must create a MONTAGE of 3–4 clips from DIFFERENT specific people.\n"
        "   → **CRITICAL**: Think about WHO would be reacting to this topic. Use your "
        "knowledge of current events and public figures to name SPECIFIC people.\n"
        "   → For political reactions: think of specific commentators, politicians, "
        "news anchors (e.g., Megyn Kelly, Ben Shapiro, Tucker Carlson, Donald Trump, "
        "AOC, Rachel Maddow, etc.)\n"
        "   → For entertainment: think of specific celebrities, YouTubers, critics.\n"
        "   → **search_query**: the primary search (e.g., 'Megyn Kelly reaction Bad Bunny halftime show')\n"
        "   → **extra_queries**: one query PER additional person, each naming a SPECIFIC person:\n"
        '     BAD:  "conservative reaction halftime show"\n'
        '     GOOD: "Megyn Kelly reaction Bad Bunny halftime show"\n'
        '     GOOD: "Ben Shapiro Bad Bunny Super Bowl rant"\n'
        '     GOOD: "Donald Trump Bad Bunny halftime Truth Social"\n'
        '     GOOD: "Fox News Bad Bunny halftime show segment"\n'
        "   → num_clips: **3–4** (each from a different person/source)\n"
        "   → Duration: **15–25 seconds total**\n\n"
        "## UNDERSTANDING THE TRANSCRIPT\n"
        "Before generating insertions, analyze:\n"
        "- What is the MAIN TOPIC? (e.g., Bad Bunny Super Bowl halftime show)\n"
        "- What SIDE or ANGLE is the host discussing? (e.g., conservative backlash)\n"
        "- WHO are the key figures involved? (e.g., Trump, Megyn Kelly, Ben Shapiro)\n"
        "- What SPECIFIC moments or clips would viewers want to see?\n"
        "- When the host says people 'lost their mind' or 'went off', WHO specifically?\n\n"
        "## RULES\n"
        "- **SCAN EVERY LINE** for cue phrases. Do NOT miss any.\n"
        "- **ONLY insert where a cue phrase exists.** No cue phrase = no clip.\n"
        "- Clip insertion starts at the EXACT timestamp of the cue phrase.\n"
        f"- Suggest AT MOST {max_clips} insertions total.\n"
        f"- Each insertion: {min_clip_seconds:.0f}–{max_clip_seconds:.0f} seconds.\n"
        "- Clips must NOT overlap.\n"
        "- Set mute=false for ALL clips (audience should hear the original audio). "
        "Only mute=true for pure music/performance clips.\n"
        "- Space insertions out; don't cluster.\n"
        "- If the narrator never cues a clip, return an empty list [].\n\n"
        "Return ONLY valid JSON (no markdown). Schema:\n"
        "[\n"
        "  {\n"
        '    "start": <float>,\n'
        '    "end": <float>,\n'
        '    "search_query": "<string — SPECIFIC person/event, not generic>",\n'
        '    "reason": "<string — must quote the cue phrase from the transcript>",\n'
        '    "clip_type": "reference" | "compilation",\n'
        '    "num_clips": <int — 1 for reference, 3-4 for compilation>,\n'
        '    "extra_queries": ["<SPECIFIC person + topic>", ...] | null,\n'
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
        model=commentary_model,
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
            mute = bool(item.get("mute", False))  # commentary clips default unmuted

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
