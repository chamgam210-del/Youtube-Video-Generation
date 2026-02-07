from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .models import TranscriptSegment


@dataclass(frozen=True)
class StorySlide:
    start: float
    end: float
    query: str


@dataclass(frozen=True)
class RichStorySlide:
    start: float
    end: float
    query: str
    headline: str
    subhead: str | None = None


@dataclass(frozen=True)
class InferredTopic:
    topic: str
    topic_type: str  # tv_show|movie|product|other


def _topic_in_query(topic: str, query: str) -> bool:
    t = (topic or "").strip().lower()
    q = (query or "").strip().lower()
    if not t or not q:
        return False
    # Loose containment is enough; we mainly want to prevent off-topic generic queries.
    return t in q


def _normalize_planned_query(*, query: str, topic: str | None, topic_shift: bool) -> str:
    q = (query or "").strip()
    if not q:
        return q

    t = (topic or "").strip()
    if t and (not topic_shift) and (not _topic_in_query(t, q)):
        # Ensure the topic is present (required by prompt spec) while keeping query short.
        q = f"{t} {q}".strip()
    return q


def pick_image_with_llm(
    *,
    slide_query: str,
    window_text: str,
    candidates: list[dict[str, Any]],
    used_source_pages: list[str],
    model: str = "gpt-4o-mini",
) -> int:
    """Pick the best candidate index for a slide.

    Returns the chosen index into `candidates`.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    if not candidates:
        raise ValueError("No candidates provided")

    # Keep payload compact: only send what the model needs to choose.
    compact = []
    for i, c in enumerate(candidates[:20]):
        compact.append(
            {
                "i": i,
                "title": c.get("title") or "",
                "source_page": c.get("page_url") or "",
                "license": c.get("license_name") or "",
                "width": int(c.get("width") or 0),
                "height": int(c.get("height") or 0),
            }
        )

    system = (
        "You are an assistant that selects the best Wikimedia Commons image candidate for a YouTube review video background. "
        "Return ONLY valid JSON (no markdown). "
        "Output schema: {\"index\": <int>} where index is one of the provided candidate i values. "
        "Choose an image that best matches the slide query and the transcript window. "
        "Avoid images whose source_page is already used if possible. "
        "Strongly prefer: real photos, on-topic stills, portraits of relevant people, or generic b-roll that matches the described scene/mood. "
        "Strongly avoid: typography/wordmarks, logos, title cards, book scans, PDF page renders, document scans, and unrelated artwork."
    )

    user = {
        "slide_query": slide_query,
        "window_text": window_text[:1200],
        "used_source_pages": used_source_pages[-20:],
        "candidates": compact,
    }

    content = _openai_chat_completions(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )

    try:
        parsed = json.loads(content)
    except Exception as e:
        raise RuntimeError(f"LLM did not return valid JSON. Got: {content[:400]}") from e

    if not isinstance(parsed, dict) or "index" not in parsed:
        raise RuntimeError(f"LLM picker returned unexpected JSON: {content[:400]}")

    idx = int(parsed["index"])
    if idx < 0 or idx >= len(candidates):
        raise RuntimeError(f"LLM picker index out of range: {idx}")
    return idx


def _openai_chat_completions(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout_s: int = 60,
) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def plan_slides_with_llm(
    segments: list[TranscriptSegment],
    *,
    audio_duration: float,
    topic: str | None,
    max_images: int,
    model: str = "gpt-4o-mini",
) -> list[StorySlide]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # Keep prompt compact: send only merged/selected segments if caller did that.
    transcript = [
        {
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "text": s.text,
        }
        for s in segments
    ]

    system = (
        "You are a video editor creating a slideshow plan for a YouTube review channel. "
        "Your job: break the transcript into a small number of background image segments, each with (start,end) and a search query. "
        "Return ONLY valid JSON (no markdown). "
        "The JSON must be an array of objects with keys: start, end, query. "
        "Optionally, objects may include: topic_shift (boolean). "
        "Times are seconds; 0 <= start < end <= audio_duration. "
        "Use at most max_images slides. "
        "Queries must be short, concrete search keywords for finding real photos/stills suitable as background visuals. "
        "The query MUST reflect what is being discussed in that exact time window (characters/actors/scenes/setting/themes), not meta commentary. "
        "Avoid channel/creator names and avoid generic queries like 'best TV shows 2025' unless the transcript is explicitly about that. "
        "Hard rule: each query MUST include the topic name unless the segment clearly shifts topics; in that case set topic_shift=true and you may omit the topic. "
        "Hard rule: query must follow one of these templates (fill placeholders as needed): "
        "1) '<TOPIC> TV series stills' "
        "2) '<TOPIC> Apple TV still' "
        "3) '<TOPIC> cast <ACTOR NAME> still' "
        "4) '<TOPIC> <CHARACTER NAME> still' "
        "5) '<TOPIC> office cubicles corporate dystopia' "
        "6) '<TOPIC> fluorescent hallway office' "
        "7) '<TOPIC> corporate office b-roll' "
        "8) '<TOPIC> retro computer terminal office' "
        "If exact stills are scarce on Commons, prefer on-theme b-roll templates (5-8) but keep the topic included. "
        "It's OK to reuse the same image for a while; don't switch too frequently."
    )

    user = {
        "audio_duration": round(audio_duration, 2),
        "topic": topic or "",
        "max_images": int(max_images),
        "transcript": transcript,
    }

    content = _openai_chat_completions(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )

    try:
        parsed = json.loads(content)
    except Exception as e:
        raise RuntimeError(f"LLM did not return valid JSON. Got: {content[:400]}") from e

    if not isinstance(parsed, list):
        raise RuntimeError("LLM JSON must be an array")

    slides: list[StorySlide] = []
    for item in parsed[: max_images]:
        if not isinstance(item, dict):
            continue
        start = float(item.get("start"))
        end = float(item.get("end"))
        query = str(item.get("query") or "").strip()
        topic_shift = bool(item.get("topic_shift") or False)
        if not query:
            continue
        if start < 0:
            start = 0.0
        if end > audio_duration:
            end = audio_duration
        if end <= start:
            continue
        query = _normalize_planned_query(query=query, topic=topic, topic_shift=topic_shift)
        slides.append(StorySlide(start=start, end=end, query=query))

    slides.sort(key=lambda s: s.start)
    # Enforce monotonic non-overlapping windows
    cleaned: list[StorySlide] = []
    cur = 0.0
    for s in slides:
        start = max(cur, s.start)
        end = max(start + 0.1, s.end)
        if start >= audio_duration:
            break
        end = min(end, audio_duration)
        cleaned.append(StorySlide(start=start, end=end, query=s.query))
        cur = end

    return cleaned[:max_images]


def classify_transcript_kind_with_llm(
    segments: list[TranscriptSegment],
    *,
    topic: str | None,
    model: str = "gpt-4o-mini",
) -> str:
    """Classify whether this transcript is a 'review' or an 'explainer'.

    Returns one of: 'review', 'explainer'.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    transcript = "\n".join(s.text.strip() for s in segments[-40:] if s.text.strip())[:4000]
    system = (
        "You classify video transcripts. Return ONLY valid JSON (no markdown). "
        "Schema: {\"kind\": \"review\"|\"explainer\"}. "
        "A review focuses on opinions/verdicts about a piece of media/product. "
        "An explainer is informational (theories, analysis topics, lists, guides) without a verdict."
    )
    user = {"topic": topic or "", "transcript": transcript}
    content = _openai_chat_completions(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )
    parsed = json.loads(content)
    kind = str((parsed or {}).get("kind") or "").strip().lower()
    if kind not in {"review", "explainer"}:
        return "review"
    return kind


def classify_transcript_kind_fallback(segments: list[TranscriptSegment] | None) -> str:
    """Heuristic classifier used when LLM isn't available."""

    if not segments:
        return "review"
    text = " ".join(s.text for s in segments[-30:]).lower()
    reviewish = any(
        w in text
        for w in (
            "review",
            "my review",
            "verdict",
            "rating",
            "stars",
            "i loved",
            "i hated",
            "overall",
            "recommend",
        )
    )
    explainerish = any(w in text for w in ("theory", "theories", "top", "rank", "explained", "breakdown", "here are"))
    if explainerish and not reviewish:
        return "explainer"
    return "review"


def infer_topic_with_llm(
    segments: list[TranscriptSegment],
    *,
    model: str = "gpt-4o-mini",
) -> InferredTopic:
    """Infer the primary subject (e.g. TV show name) from transcript."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    transcript = "\n".join(s.text.strip() for s in segments[-50:] if s.text.strip())[:5000]
    system = (
        "You infer the primary topic of a narration. Return ONLY valid JSON (no markdown). "
        "Schema: {\"topic\": <string>, \"topic_type\": \"tv_show\"|\"movie\"|\"product\"|\"other\"}. "
        "topic must be short (2-6 words), just the name (no quotes), e.g. 'Severance'."
    )
    user = {"transcript": transcript}
    content = _openai_chat_completions(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )
    parsed = json.loads(content)
    topic = str((parsed or {}).get("topic") or "").strip()
    topic_type = str((parsed or {}).get("topic_type") or "").strip().lower()
    if not topic:
        topic = ""
    if topic_type not in {"tv_show", "movie", "product", "other"}:
        topic_type = "other"
    return InferredTopic(topic=topic, topic_type=topic_type)


def infer_topic_fallback(*, audio_stem: str) -> InferredTopic:
    stem = (audio_stem or "").replace("_", " ").replace("-", " ").strip()
    stem = " ".join(stem.split())
    # Very lightweight: last 1-2 words are often the actual subject.
    parts = stem.split()
    guess = " ".join(parts[-2:]) if len(parts) >= 2 else (parts[0] if parts else "")
    return InferredTopic(topic=guess, topic_type="other")


def plan_rich_slides_with_llm(
    segments: list[TranscriptSegment],
    *,
    audio_duration: float,
    topic: str | None,
    max_slides: int,
    kind: str,  # explainer|shorts
    model: str = "gpt-4o-mini",
) -> list[RichStorySlide]:
    """Plan a storyboard that includes on-screen text + image queries."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    transcript = [
        {
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "text": s.text,
        }
        for s in segments
    ]

    k = (kind or "").strip().lower()
    if k not in {"explainer", "shorts"}:
        k = "explainer"

    system = (
        "You are a video editor creating a storyboard plan for a narrated slideshow video. "
        "Return ONLY valid JSON (no markdown). "
        "JSON must be an array of objects with keys: start, end, query, headline. "
        "Optional: subhead. "
        "Times are seconds; 0 <= start < end <= audio_duration. "
        "Use at most max_slides slides. "
        "query must be short search keywords for finding real photos/stills suitable as background visuals. "
        "headline must be a short on-screen caption (max 8 words) that matches what is being said in that time window. "
        "If the content is list-like (e.g. theories), each slide can represent one item. "
        "If topic is empty, infer the primary topic (show/product) from transcript and use it consistently in queries. "
        "If the transcript is about a TV series, queries MUST be about that TV series and should include keywords like 'TV series still', 'cast', or 'scene still'. "
        "If the provided topic hint conflicts with transcript, ignore the topic hint. "
        "Avoid spoilers if this is media-related."
    )
    if k == "shorts":
        system += (
            " This is for a YouTube Short: keep slides punchy; prefer 4-10 slides; "
            "headline should be very short (max 6 words) and hooky."
        )

    user = {
        "kind": k,
        "audio_duration": round(float(audio_duration), 2),
        "topic": topic or "",
        "max_slides": int(max_slides),
        "transcript": transcript,
    }

    content = _openai_chat_completions(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )

    parsed = json.loads(content)
    if not isinstance(parsed, list):
        raise RuntimeError("LLM JSON must be an array")

    slides: list[RichStorySlide] = []
    for item in parsed[: max_slides]:
        if not isinstance(item, dict):
            continue
        start = float(item.get("start"))
        end = float(item.get("end"))
        query = str(item.get("query") or "").strip()
        headline = str(item.get("headline") or "").strip()
        subhead = str(item.get("subhead") or "").strip() if item.get("subhead") else None
        if not query or not headline:
            continue
        if start < 0:
            start = 0.0
        if end > audio_duration:
            end = audio_duration
        if end <= start:
            continue
        slides.append(RichStorySlide(start=start, end=end, query=query, headline=headline, subhead=subhead))

    slides.sort(key=lambda s: s.start)
    cleaned: list[RichStorySlide] = []
    cur = 0.0
    for s in slides:
        start = max(cur, s.start)
        end = max(start + 0.1, s.end)
        if start >= audio_duration:
            break
        end = min(end, audio_duration)
        cleaned.append(RichStorySlide(start=start, end=end, query=s.query, headline=s.headline, subhead=s.subhead))
        cur = end

    return cleaned[:max_slides]


def generate_video_title_with_llm(
    segments: list[TranscriptSegment],
    *,
    topic: str | None,
    channel_name: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Generate a YouTube-friendly title based on the transcript.

    Returns a plain string. Raises if OPENAI_API_KEY is missing.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # Keep payload compact.
    joined = " ".join((s.text or "").strip() for s in segments if (s.text or "").strip())
    transcript_snippet = joined[:4000]

    system = (
        "You are a YouTube video producer for a review channel. "
        "Write ONE punchy, click-worthy title based on the provided transcript snippet. "
        "Return ONLY valid JSON (no markdown). Schema: {\"title\": <string>}. "
        "Hard rules: "
        "- Keep it under 70 characters. "
        "- Do NOT include the channel name. "
        "- Avoid profanity; keep it advertiser-safe. "
        "- No hashtags, no quotes around the whole title, no emojis. "
        "- It must reflect the transcript's actual stance/themes." 
    )

    user = {
        "channel_name": channel_name,
        "topic_hint": topic or "",
        "transcript_snippet": transcript_snippet,
    }

    content = _openai_chat_completions(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )

    try:
        parsed = json.loads(content)
    except Exception as e:
        raise RuntimeError(f"LLM did not return valid JSON. Got: {content[:400]}") from e

    title = ""
    if isinstance(parsed, dict):
        title = str(parsed.get("title") or "").strip()

    # Final guardrails.
    if channel_name and channel_name.lower() in title.lower():
        title = title.replace(channel_name, "").strip(" -|:")

    return title


def generate_video_title_fallback(audio_path: str | Path, *, topic: str | None) -> str:
    t = (topic or "").strip()
    if t:
        return f"{t} Review"

    p = Path(audio_path)
    stem = p.stem.replace("_", " ").strip()
    return stem[:80] if stem else "Brutally Honest Review"
