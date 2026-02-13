from __future__ import annotations

import json
import os
import re
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
class HighlightClip:
    start: float
    end: float
    reason: str = ""


@dataclass(frozen=True)
class HighlightSpan:
    start_i: int
    end_i: int
    reason: str = ""


@dataclass(frozen=True)
class InferredTopic:
    topic: str
    topic_type: str  # tv_show|movie|product|other


@dataclass(frozen=True)
class ShortsReviewHookFrame:
    image_type: str  # poster|closeup|prestige_still|reaction_closeup|neutral
    text: str
    duration: float
    motion: str  # none


@dataclass(frozen=True)
class ShortsReviewBeat:
    timestamp: str
    line: str
    image_type: str  # prestige_still|reaction_closeup|closeup|neutral|poster
    motion: str  # none|slow_zoom|slow_zoom_in|snap_zoom|minimal
    text: str
    priority: str  # hook|tension|payoff|verdict|loop
    importance: str  # low|medium|high


@dataclass(frozen=True)
class ShortsReviewEndingFrame:
    image_type: str  # neutral|poster|closeup
    text: str
    duration: float
    motion: str  # none


@dataclass(frozen=True)
class ShortsReviewStoryboard:
    hook_frame: ShortsReviewHookFrame
    beats: list[ShortsReviewBeat]
    ending_frame: ShortsReviewEndingFrame


_TS_RE = re.compile(r"(?P<s>\d+(?:\.\d+)?)\s*(?:–|—|-|to)\s*(?P<e>\d+(?:\.\d+)?)")

_PIVOT_RE = re.compile(r"\b(but|however|here'?s the thing)\b", re.IGNORECASE)


def _parse_timestamp_range(ts: str) -> tuple[float, float] | None:
    m = _TS_RE.search(str(ts or ""))
    if not m:
        return None
    try:
        s = float(m.group("s"))
        e = float(m.group("e"))
    except Exception:
        return None
    if e <= s:
        return None
    return s, e


def _is_pivot(line: str) -> bool:
    return bool(_PIVOT_RE.search(str(line or "")))


def _normalize_hook_text(t: str) -> str:
    txt = " ".join(str(t or "").replace("\n", " ").split()).strip()
    if not txt:
        return "HOW?!"

    low = txt.lower()
    banned = {
        "this movie",
        "this show",
        "this film",
        "my review",
        "review",
    }
    if any(b in low for b in banned):
        return "HOW?!"

    # Encourage curiosity punctuation.
    if ("?" not in txt) and ("!" not in txt):
        txt = txt.upper() + "?"
    return txt


def _two_words_max(text: str) -> str:
    t = " ".join(str(text or "").replace("\n", " ").split()).strip()
    if not t:
        return ""
    # Keep emojis/punctuation if it is already a short CTA.
    words = [w for w in t.split(" ") if w]
    if len(words) <= 2:
        return t
    return " ".join(words[:2]).strip()


def _coerce_choice(val: str, allowed: set[str], default: str) -> str:
    v = str(val or "").strip().lower()
    return v if v in allowed else default


def parse_shorts_review_storyboard(obj: Any) -> ShortsReviewStoryboard:
    if not isinstance(obj, dict):
        raise ValueError("Story schema must be a JSON object")

    hook = obj.get("hook_frame")
    beats = obj.get("beats")
    ending = obj.get("ending_frame")

    if not isinstance(hook, dict):
        raise ValueError("hook_frame is required")
    if not isinstance(beats, list):
        raise ValueError("beats must be an array")
    if not isinstance(ending, dict):
        raise ValueError("ending_frame is required")

    hook_frame = ShortsReviewHookFrame(
        image_type=_coerce_choice(
            hook.get("image_type"),
            {"poster", "closeup", "prestige_still", "reaction_closeup", "neutral"},
            "poster",
        ),
        text=str(hook.get("text") or "").strip(),
        duration=float(hook.get("duration") or 1.6),
        motion=_coerce_choice(hook.get("motion"), {"none"}, "none"),
    )

    ending_frame = ShortsReviewEndingFrame(
        image_type=_coerce_choice(
            ending.get("image_type"),
            {"neutral", "poster", "closeup", "prestige_still", "reaction_closeup"},
            "neutral",
        ),
        text=str(ending.get("text") or "").strip(),
        duration=float(ending.get("duration") or 1.2),
        motion=_coerce_choice(ending.get("motion"), {"none"}, "none"),
    )

    out_beats: list[ShortsReviewBeat] = []
    for item in beats:
        if not isinstance(item, dict):
            continue

        ts = str(item.get("timestamp") or "").strip()
        parsed = _parse_timestamp_range(ts)
        if not parsed:
            # Accept start/end numeric as a back-compat format.
            try:
                s = float(item.get("start"))
                e = float(item.get("end"))
                if e > s:
                    ts = f"{s:.2f}-{e:.2f}"
            except Exception:
                ts = ""
        else:
            # Timestamp clamp: keep beats short so captions don't linger.
            s, e = float(parsed[0]), float(parsed[1])
            if (e - s) > 2.2:
                e = s + 2.2
            ts = f"{s:.2f}-{e:.2f}"

        out_beats.append(
            ShortsReviewBeat(
                timestamp=ts,
                line=str(item.get("line") or "").strip(),
                image_type=_coerce_choice(
                    item.get("image_type"),
                    {"poster", "closeup", "prestige_still", "reaction_closeup", "neutral"},
                    "prestige_still",
                ),
                motion=_coerce_choice(
                    item.get("motion"),
                    {"none", "slow_zoom", "slow_zoom_in", "snap_zoom", "minimal"},
                    "slow_zoom",
                ),
                text=str(item.get("text") or "").strip(),
                priority=_coerce_choice(
                    item.get("priority"),
                    {"hook", "tension", "payoff", "verdict", "loop"},
                    "tension",
                ),
                importance=_coerce_choice(item.get("importance"), {"low", "medium", "high"}, "medium"),
            )
        )

    return ShortsReviewStoryboard(hook_frame=hook_frame, beats=out_beats, ending_frame=ending_frame)


def normalize_shorts_review_storyboard(
    sb: ShortsReviewStoryboard,
    *,
    hook_seconds: float,
    ending_seconds: float,
    max_beats: int,
) -> ShortsReviewStoryboard:
    hook_s = max(0.0, float(hook_seconds))
    end_s = max(0.0, float(ending_seconds))

    # Enforce hook/ending invariants.
    hook = ShortsReviewHookFrame(
        image_type=sb.hook_frame.image_type,
        text=_normalize_hook_text(sb.hook_frame.text),
        duration=hook_s,
        motion="none",
    )
    ending = ShortsReviewEndingFrame(
        # Loop-compat: keep ending background compatible with the hook palette/composition.
        image_type=(hook.image_type or sb.ending_frame.image_type),
        text=(str(sb.ending_frame.text or "").strip() or "AGREE? 👇"),
        duration=end_s,
        motion="none",
    )

    beats = sb.beats[: max(0, int(max_beats))]

    # Hook contrast (Rule A): hook must visually reset vs beat 1.
    # We can't measure brightness/crop/temperature directly, so we enforce a reliable proxy:
    # - If hook is a poster (wide/centered), beat 1 becomes a close-up (subject distance + crop change).
    # - If hook is a close-up, beat 1 becomes a poster (subject distance + composition reset).
    if beats:
        desired_first = beats[0].image_type
        hook_it = (hook.image_type or "").strip().lower()
        if hook_it == "poster":
            if beats[0].image_type not in {"closeup", "reaction_closeup"}:
                desired_first = "closeup"
        elif hook_it in {"closeup", "reaction_closeup"}:
            desired_first = "poster"
        else:
            if beats[0].image_type == hook.image_type:
                desired_first = "closeup" if hook_it == "poster" else "prestige_still"

        if beats[0].image_type != desired_first:
            beats[0] = ShortsReviewBeat(
                timestamp=beats[0].timestamp,
                line=beats[0].line,
                image_type=desired_first,
                motion=beats[0].motion,
                text=beats[0].text,
                priority=beats[0].priority,
                importance=beats[0].importance,
            )

    # Caption density: <=2 words except hook/ending (we keep those as-is).
    cleaned: list[ShortsReviewBeat] = []
    for b in beats:
        txt = _two_words_max(b.text)
        # Pivot enforcement (Rule B): do NOT carry "BUT…" forward on the spoken beat.
        # The pipeline inserts a dedicated pivot interrupt card + silence before the words.
        if _is_pivot(b.line):
            txt = ""
        cleaned.append(
            ShortsReviewBeat(
                timestamp=b.timestamp,
                line=b.line,
                image_type=b.image_type,
                motion=b.motion,
                text=txt,
                priority=("tension" if _is_pivot(b.line) else b.priority),
                importance=("high" if _is_pivot(b.line) else b.importance),
            )
        )

    # Caption integrity: if text doesn't change, image shouldn't change.
    # Collapse consecutive beats with identical text (including empty) into one visual beat.
    def _merge_ts(a: str, b: str) -> str:
        ra = _parse_timestamp_range(a)
        rb = _parse_timestamp_range(b)
        if not ra or not rb:
            return a
        s = float(ra[0])
        e = float(rb[1])
        if e <= s:
            return a
        return f"{s:.2f}-{e:.2f}"

    collapsed: list[ShortsReviewBeat] = []
    for b in cleaned:
        if collapsed and (collapsed[-1].text == b.text):
            prev = collapsed[-1]
            collapsed[-1] = ShortsReviewBeat(
                timestamp=_merge_ts(prev.timestamp, b.timestamp),
                line=(prev.line + " " + b.line).strip(),
                image_type=prev.image_type,
                motion=prev.motion,
                text=prev.text,
                priority=prev.priority,
                importance=prev.importance,
            )
        else:
            collapsed.append(b)

    cleaned = collapsed

    # Visual diversity: avoid repeating the same image_type back-to-back.
    for i in range(1, len(cleaned)):
        prev = cleaned[i - 1]
        cur = cleaned[i]
        if prev.image_type == cur.image_type:
            alt = "closeup"
            if cur.image_type in {"closeup", "reaction_closeup"}:
                alt = "prestige_still"
            elif cur.image_type == "prestige_still":
                alt = "reaction_closeup"
            elif cur.image_type == "poster":
                alt = "closeup"
            cleaned[i] = ShortsReviewBeat(
                timestamp=cur.timestamp,
                line=cur.line,
                image_type=alt,
                motion=cur.motion,
                text=cur.text,
                priority=cur.priority,
                importance=cur.importance,
            )

    # Importance hierarchy: cap highs to 3, keep at least 2 when possible.
    highs = [i for i, b in enumerate(cleaned) if b.importance == "high"]
    if len(highs) > 3:
        # Downgrade later highs first.
        for i in highs[3:]:
            b = cleaned[i]
            cleaned[i] = ShortsReviewBeat(
                timestamp=b.timestamp,
                line=b.line,
                image_type=b.image_type,
                motion=b.motion,
                text=b.text,
                priority=b.priority,
                importance="medium",
            )
    elif len(highs) < 2 and len(cleaned) >= 2:
        # Promote hook/tension pivots when the model didn't pick any.
        for i, b in enumerate(cleaned[:6]):
            if b.priority in {"hook", "payoff", "verdict"}:
                cleaned[i] = ShortsReviewBeat(
                    timestamp=b.timestamp,
                    line=b.line,
                    image_type=b.image_type,
                    motion=b.motion,
                    text=b.text,
                    priority=b.priority,
                    importance="high",
                )
                highs.append(i)
                if len([x for x in highs if x == i]) >= 2:
                    break

    # Motion by intent (never random).
    mapped: list[ShortsReviewBeat] = []
    for b in cleaned:
        pr = b.priority
        if pr == "hook":
            motion = "none"
        elif pr == "tension":
            motion = "snap_zoom"
        elif pr == "payoff":
            motion = "minimal"
        elif pr == "loop":
            motion = "none"
        else:  # verdict
            motion = "minimal"

        mapped.append(
            ShortsReviewBeat(
                timestamp=b.timestamp,
                line=b.line,
                image_type=b.image_type,
                motion=motion,
                text=b.text,
                priority=b.priority,
                importance=b.importance,
            )
        )

    # Reduce motion fatigue: cap snap_zoom usage (motion is punctuation, not decoration).
    snap_cap_ratio = 0.45
    snaps = 0
    out_mapped: list[ShortsReviewBeat] = []
    n = max(1, len(mapped))
    for b in mapped:
        m = b.motion
        if m == "snap_zoom":
            snaps += 1
            if (snaps / float(n)) > float(snap_cap_ratio):
                m = "minimal"
        out_mapped.append(
            ShortsReviewBeat(
                timestamp=b.timestamp,
                line=b.line,
                image_type=b.image_type,
                motion=m,
                text=b.text,
                priority=b.priority,
                importance=b.importance,
            )
        )

    mapped = out_mapped

    return ShortsReviewStoryboard(hook_frame=hook, beats=mapped, ending_frame=ending)


def plan_shorts_review_storyboard_with_llm(
    segments: list[TranscriptSegment],
    *,
    audio_duration: float,
    topic: str | None,
    max_beats: int,
    hook_seconds: float = 4.0,
    ending_seconds: float = 4.0,
    model: str = "gpt-4o-mini",
) -> ShortsReviewStoryboard:
    """Generate a retention-first Shorts Review storyboard with a strict schema.

    Timestamps are in VIDEO time (i.e. transcript times are shifted by hook_seconds).
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    hook_s = max(0.0, float(hook_seconds))
    end_s = max(0.0, float(ending_seconds))
    total = hook_s + max(0.0, float(audio_duration))

    transcript = []
    for s in segments:
        try:
            st = round(float(s.start) + hook_s, 2)
            en = round(float(s.end) + hook_s, 2)
        except Exception:
            continue
        if en <= st:
            continue
        transcript.append({"start": st, "end": en, "text": str(s.text or "").strip()})

    system = (
        "You are generating a storyboard for a YouTube Short (Shorts Review). "
        "Return ONLY valid JSON (no markdown). "
        "You MUST output valid JSON following the provided schema. "
        "\n\nRules you MUST follow:\n"
        "- The first 1.6 seconds must be a static hook frame designed to stop scrolling.\n"
        "- Rule A (Hook contrast): hook frame must differ from beat 1 in 2+ visual dimensions: brightness, crop, color temperature, subject distance.\n"
        "  (Example: hook = dark centered poster; beat 1 = brighter off-center close-up.)\n"
        "- Visuals must change every 1–2 seconds (never hold long).\n"
        "- Rule B (Pivot pause): pivot phrases (but/however/here's the thing) must get snap zoom AND a 0.25–0.35s silence pause before the words.\n"
        "  Pivot beat timestamps MUST start exactly at the pivot phrase.\n"
        "- Do NOT evenly distribute timing.\n"
        "- The storyboard must delay the verdict and create curiosity.\n"
        "- The final frame must encourage looping (static, question-based).\n"
        "- The ending frame must be visually compatible with the hook (similar palette/contrast/simplicity) so the loop seam is less noticeable.\n"
        "- On-screen text must be <= 2 words per beat (EXCEPT hook_frame and ending_frame).\n"
        "- Rule C (Caption clamp): beat timestamps must closely match spoken line duration; text must not outlive the spoken line by >0.1s.\n"
        "- Only 2–3 beats may have importance='high'.\n"
        "\nMotion must be chosen by intent (never random):\n"
        "- hook -> motion none\n"
        "- tension -> motion snap_zoom\n"
        "- payoff/verdict -> motion minimal\n"
        "- loop -> motion none\n"
        "\nDo NOT describe scenes literally. Focus on emotion, contrast, and retention."
    )

    schema = {
        "hook_frame": {"image_type": "poster|closeup", "text": "OSCAR BAIT?", "duration": 1.6, "motion": "none"},
        "beats": [
            {
                "timestamp": "1.6-3.6",
                "line": "This movie got Oscar nominations.",
                "image_type": "prestige_still|reaction_closeup|closeup|neutral",
                "motion": "slow_zoom|snap_zoom|minimal|none",
                "text": "OSCAR-NOM",
                "priority": "hook|tension|payoff|verdict|loop",
                "importance": "low|medium|high",
            }
        ],
        "ending_frame": {"image_type": "neutral", "text": "AGREE? 👇", "duration": 1.2, "motion": "none"},
    }

    user = {
        "topic": (topic or "").strip(),
        "video_duration": round(float(total), 2),
        "hook_seconds": round(float(hook_s), 2),
        "ending_seconds": round(float(end_s), 2),
        "max_beats": int(max_beats),
        "transcript": transcript,
        "schema": schema,
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
    sb = parse_shorts_review_storyboard(parsed)
    return normalize_shorts_review_storyboard(sb, hook_seconds=hook_s, ending_seconds=end_s, max_beats=max_beats)


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


def suggest_image_search_queries_with_llm(
    *,
    anchor: str,
    window_text: str,
    topic_type: str | None = None,
    video_type: str | None = None,
    model: str = "gpt-4o-mini",
    max_queries: int = 5,
) -> list[str]:
    """Suggest search queries for finding relevant stills.

    The goal is to keep results within the same movie/show (anchor), but vary by transcript context.
    Returns a list of short queries (strings). Raises if OPENAI_API_KEY is missing.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    a = (anchor or "").strip()
    wt = (window_text or "").strip()
    if not a or not wt:
        return []

    n = int(max(1, min(8, max_queries)))

    system = (
        "You generate web image search queries for finding stills from a specific movie or TV show. "
        "Return ONLY valid JSON (no markdown). "
        "Output schema: {\"queries\": [<string>, ...]} with 3 to 5 items. "
        "Every query MUST include the exact anchor string (the movie/show name). "
        "Use the transcript window to pick concrete visual keywords (e.g., guitar, band, soundtrack, performance, tense scene). "
        "Keep queries short (4-9 words). "
        "Prefer scene imagery terms like: scene still, screencap, frame, close up. "
        "Avoid: review site names, years unless already in anchor, 'poster', 'official poster', 'logo'."
    )

    user = {
        "anchor": a,
        "topic_type": (topic_type or "").strip(),
        "video_type": (video_type or "").strip(),
        "window_text": wt[:900],
        "max_queries": n,
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

    qs = parsed.get("queries") if isinstance(parsed, dict) else None
    if not isinstance(qs, list):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for q in qs:
        if not isinstance(q, str):
            continue
        s = " ".join(q.split()).strip()
        if not s:
            continue
        sl = s.lower()
        if "poster" in sl or "logo" in sl:
            continue
        if a.lower() not in sl:
            # Enforce anchoring defensively.
            s = f"{a} {s}".strip()
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= n:
            break

    return out


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


def _clean_highlight_clips(
    clips: list[HighlightClip],
    *,
    audio_duration: float,
    max_clips: int,
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> list[HighlightClip]:
    if not clips:
        return []

    dur = max(0.0, float(audio_duration))
    out: list[HighlightClip] = []
    for c in clips:
        try:
            s = max(0.0, min(dur, float(c.start)))
            e = max(0.0, min(dur, float(c.end)))
        except Exception:
            continue
        if e <= s:
            continue

        # Clamp length.
        length = e - s
        if length < float(min_clip_seconds):
            e = min(dur, s + float(min_clip_seconds))
        elif length > float(max_clip_seconds):
            e = s + float(max_clip_seconds)

        if e <= s:
            continue
        out.append(HighlightClip(start=s, end=e, reason=str(c.reason or "").strip()))

    out.sort(key=lambda x: float(x.start))

    # Remove overlaps by trimming later clips.
    cleaned: list[HighlightClip] = []
    cur = 0.0
    for c in out:
        s = max(cur, float(c.start))
        e = float(c.end)
        if e <= s:
            continue
        cleaned.append(HighlightClip(start=s, end=e, reason=c.reason))
        cur = float(e)
        if len(cleaned) >= int(max_clips):
            break

    return cleaned


def _detect_spoiler_warning_clip(
    segments: list[TranscriptSegment],
    *,
    audio_duration: float,
    detect_within_seconds: float = 60.0,
    detect_jitter_seconds: float = 5.0,
    target_clip_seconds: float = 40.0,
) -> HighlightClip | None:
    """Detect a spoken spoiler warning early in the review.

    If detected, return a single clip from t=0 to shortly after the warning,
    so the Shorts cut stays coherent and ends right after the warning is said.
    """

    dur = max(0.0, float(audio_duration))
    if dur <= 0.0:
        return None

    # User intent: if a spoiler warning is detected by ~60s, make a short that ends
    # right after the warning. Allow small jitter since timestamps can drift.
    scan_until = min(dur, float(detect_within_seconds) + float(detect_jitter_seconds))
    if scan_until <= 0.0:
        return None

    def _norm(txt: str) -> str:
        return " ".join(str(txt or "").lower().replace("-", " ").split())

    def _is_negative_context(txt: str) -> bool:
        # Avoid false positives like "spoiler-free" or "no spoilers".
        return (
            "spoiler free" in txt
            or "spoiler-free" in txt
            or "spoilerfree" in txt
            or "no spoilers" in txt
            or "without spoilers" in txt
        )

    def _is_warning(txt: str) -> bool:
        if _is_negative_context(txt):
            return False
        if "spoiler warning" in txt or "spoiler alert" in txt:
            return True
        if "spoiler" in txt and ("warning" in txt or "alert" in txt):
            return True
        # Common phrasing: "from here on, spoilers" / "from here it's spoilers".
        if ("from here" in txt or "from now" in txt) and ("spoiler" in txt or "spoilers" in txt):
            return True
        return False

    # Find earliest warning segment within the first minute.
    idx = None
    for i, seg in enumerate(segments[:300]):
        try:
            if float(seg.start) > scan_until:
                break
        except Exception:
            continue
        txt = _norm(getattr(seg, "text", ""))
        if not txt:
            continue
        if _is_warning(txt):
            idx = i
            break

    if idx is None:
        return None

    # Include the entire warning utterance, possibly spanning adjacent segments.
    warning_end = 0.0
    try:
        warning_end = float(segments[idx].end)
    except Exception:
        warning_end = 0.0

    j = idx + 1
    while j < len(segments):
        try:
            if float(segments[j].start) > scan_until:
                break
        except Exception:
            break

        gap = 999.0
        try:
            gap = float(segments[j].start) - float(segments[j - 1].end)
        except Exception:
            gap = 999.0

        nxt = _norm(getattr(segments[j], "text", ""))
        if gap <= 0.75 and ("spoiler" in nxt or "warning" in nxt or "alert" in nxt):
            try:
                warning_end = max(warning_end, float(segments[j].end))
            except Exception:
                pass
            j += 1
            continue
        break

    # End should include the spoiler warning being spoken.
    end_after_warning = min(dur, max(0.0, warning_end + 0.50))
    if end_after_warning <= 0.25:
        return None

    # User request: the Shorts cut should literally be the beginning of the full review.
    # If a spoiler warning is detected early (within ~1 minute), cap the intro cut at ~40s.
    start = 0.0
    cutoff = float(target_clip_seconds)
    if cutoff > 0:
        end = min(dur, cutoff)
    else:
        # Fallback: end shortly after warning (shouldn't happen with default settings).
        end = float(end_after_warning)

    if end <= 0.25:
        return None

    return HighlightClip(start=float(start), end=float(end), reason="Spoiler warning")


def _looks_like_sentence_start(txt: str) -> bool:
    t = (txt or "").strip()
    if not t:
        return True
    # If it starts with a lowercase letter or a connective, it likely continues a thought.
    if t[:1].isalpha() and t[:1].islower():
        return False
    return True


def _looks_like_sentence_end(txt: str) -> bool:
    t = (txt or "").strip()
    if not t:
        return True
    return t.endswith((".", "!", "?"))


def _expand_span_to_sentence_boundaries(
    segments: list[TranscriptSegment], *, start_i: int, end_i: int, max_expand_seconds: float = 7.0
) -> tuple[int, int]:
    n = len(segments)
    if n == 0:
        return start_i, end_i
    s = max(0, min(n - 1, int(start_i)))
    e = max(s, min(n - 1, int(end_i)))

    # Expand backwards if the first text looks mid-thought.
    try:
        base_start = float(segments[s].start)
    except Exception:
        base_start = 0.0
    while s > 0 and (not _looks_like_sentence_start(segments[s].text)):
        try:
            prev_start = float(segments[s - 1].start)
        except Exception:
            prev_start = base_start
        if (base_start - prev_start) > float(max_expand_seconds):
            break
        s -= 1
        try:
            base_start = float(segments[s].start)
        except Exception:
            break

    # Expand forwards if the last text looks like it cuts off.
    try:
        base_end = float(segments[e].end)
    except Exception:
        base_end = 0.0
    while e < (n - 1) and (not _looks_like_sentence_end(segments[e].text)):
        try:
            next_end = float(segments[e + 1].end)
        except Exception:
            next_end = base_end
        if (next_end - base_end) > float(max_expand_seconds):
            break
        e += 1
        try:
            base_end = float(segments[e].end)
        except Exception:
            break

    return s, e


def _token_jaccard(a: str, b: str) -> float:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "so",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "it",
        "this",
        "that",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
    }

    def toks(s: str) -> set[str]:
        raw = "".join(ch.lower() if ch.isalnum() else " " for ch in (s or ""))
        out = {t for t in raw.split() if len(t) >= 3 and t not in stop}
        return out

    ta = toks(a)
    tb = toks(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return float(inter) / float(union) if union else 0.0


def _clips_from_spans(
    segments: list[TranscriptSegment],
    spans: list[HighlightSpan],
    *,
    audio_duration: float,
    max_total_seconds: float = 60.0,
    min_clip_seconds: float = 5.0,
    max_clip_seconds: float = 20.0,
) -> list[HighlightClip]:
    dur = max(0.0, float(audio_duration))
    if dur <= 0.0 or not segments or not spans:
        return []

    # Sort spans chronologically by segment start.
    spans2 = []
    for sp in spans:
        try:
            s_i = int(sp.start_i)
            e_i = int(sp.end_i)
        except Exception:
            continue
        if e_i < s_i:
            s_i, e_i = e_i, s_i
        s_i = max(0, min(len(segments) - 1, s_i))
        e_i = max(0, min(len(segments) - 1, e_i))
        spans2.append(HighlightSpan(start_i=s_i, end_i=e_i, reason=str(sp.reason or "").strip()))

    spans2.sort(key=lambda sp: float(segments[sp.start_i].start))

    picked: list[HighlightClip] = []
    used_text = ""
    total = 0.0
    cap = min(60.0, max(10.0, float(max_total_seconds)))

    for sp in spans2:
        s_i, e_i = _expand_span_to_sentence_boundaries(segments, start_i=sp.start_i, end_i=sp.end_i)

        try:
            start = float(segments[s_i].start)
            end = float(segments[e_i].end)
        except Exception:
            continue

        start = max(0.0, min(dur, start))
        end = max(0.0, min(dur, end))
        if end <= start:
            continue

        # Enforce per-clip max length.
        if (end - start) > float(max_clip_seconds):
            end = start + float(max_clip_seconds)

        # Deduplicate: skip clips that repeat the same point.
        clip_text = " ".join((segments[i].text or "").strip() for i in range(s_i, min(e_i + 1, len(segments))))
        if used_text:
            if _token_jaccard(used_text, clip_text) >= 0.72:
                continue

        # Budget.
        remaining = cap - total
        if remaining <= 0.0:
            break

        if (end - start) > remaining:
            end = start + remaining

        if (end - start) < float(min_clip_seconds):
            continue

        picked.append(HighlightClip(start=start, end=end, reason=sp.reason))
        total += float(end - start)
        used_text = (used_text + " " + clip_text).strip()

    # Final: de-overlap + clamp.
    return _clean_highlight_clips(
        picked,
        audio_duration=dur,
        max_clips=max(12, len(picked) or 0),
        min_clip_seconds=float(min_clip_seconds),
        max_clip_seconds=float(max_clip_seconds),
    )


def extract_review_highlights_meaningful_with_llm(
    segments: list[TranscriptSegment],
    *,
    audio_duration: float,
    max_total_seconds: float = 60.0,
    model: str = "gpt-4o-mini",
) -> list[HighlightClip]:
    """Create a meaningful <=60s highlight montage from a review transcript.

    The model selects segment-index spans (not raw seconds) so we can cut on transcript
    boundaries and avoid mid-sentence jumps.
    """

    if not segments:
        return []

    dur = max(0.0, float(audio_duration))
    if dur <= 0.0:
        return []

    # Keep the existing spoiler-intro behavior.
    spoiler_clip = _detect_spoiler_warning_clip(segments, audio_duration=dur)
    if spoiler_clip is not None:
        return [spoiler_clip]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    transcript = []
    for i, s in enumerate(segments[:320]):
        transcript.append(
            {
                "i": int(i),
                "start": round(float(s.start), 2),
                "end": round(float(s.end), 2),
                "text": str(s.text or "")[:260],
            }
        )

    system = (
        "You are a senior video editor. Build a 1-minute MAX highlight montage from a FULL review transcript. "
        "Your cut must sound like a coherent short: no mid-sentence cuts, no abrupt topic whiplash, and no repeated points. "
        "If the reviewer repeats themselves, pick only the strongest occurrence. "
        "Return ONLY valid JSON (no markdown). "
        "Schema: {\"spans\": [{\"start_i\": int, \"end_i\": int, \"reason\": string}]}. "
        "Rules: spans must be in chronological order, not overlap, and each span should start/end on segment boundaries. "
        "Each span should be roughly 6-20 seconds and the total combined duration must be <= max_total_seconds (hard cap 60). "
        "Each span must begin with a complete sentence that can stand alone (no 'and/but/so' mid-thought starts). "
        "Avoid duplicate points even if paraphrased. "
        "Pick spans that cover: hook, core opinion(s), key pros/cons, and a final takeaway." 
    )

    user = {
        "audio_duration": round(dur, 2),
        "max_total_seconds": float(min(60.0, float(max_total_seconds))),
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

    raw_spans = (parsed or {}).get("spans") if isinstance(parsed, dict) else None
    if not isinstance(raw_spans, list):
        raise RuntimeError("LLM JSON must be an object with key 'spans' (array)")

    spans: list[HighlightSpan] = []
    for item in raw_spans:
        if not isinstance(item, dict):
            continue
        try:
            s_i = int(item.get("start_i"))
            e_i = int(item.get("end_i"))
        except Exception:
            continue
        spans.append(HighlightSpan(start_i=s_i, end_i=e_i, reason=str(item.get("reason") or "").strip()))

    return _clips_from_spans(
        segments,
        spans,
        audio_duration=dur,
        max_total_seconds=float(min(60.0, float(max_total_seconds))),
        min_clip_seconds=6.0,
        max_clip_seconds=20.0,
    )


def extract_review_highlights_with_llm(
    segments: list[TranscriptSegment],
    *,
    audio_duration: float,
    max_clips: int = 4,
    target_total_seconds: float = 35.0,
    min_clip_seconds: float = 6.0,
    max_clip_seconds: float = 14.0,
    model: str = "gpt-4o-mini",
) -> list[HighlightClip]:
    """Pick the most important moments from a review transcript.

    Returns a small set of timestamped clips (start/end in original audio seconds)
    intended to be concatenated into a Shorts-length highlight cut.
    """
    if not segments:
        return []

    dur = max(0.0, float(audio_duration))
    if dur <= 0.0:
        return []

    # Special case: if we detect a spoiler warning early, the Shorts should be
    # the intro up to (and including) the spoiler warning, then stop.
    spoiler_clip = _detect_spoiler_warning_clip(segments, audio_duration=dur)
    if spoiler_clip is not None:
        return [spoiler_clip]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # Keep prompt compact: use a limited window of segments (usually merged by caller).
    transcript = [
        {
            "start": round(float(s.start), 2),
            "end": round(float(s.end), 2),
            "text": str(s.text or "")[:280],
        }
        for s in segments[:250]
    ]

    system = (
        "You are a video editor creating a YouTube Shorts highlight cut from a FULL-LENGTH review. "
        "Pick the MOST IMPORTANT and MOST INTERESTING moments that represent the core points and verdict. "
        "CRITICAL: the highlight must feel coherent when watched as a single short. "
        "Prefer ONE contiguous excerpt (a single clip) that stands on its own, instead of stitching unrelated parts. "
        "If you return multiple clips, they must be in strict chronological order and should be near-adjacent (small gaps) so the cut makes sense. "
        "Avoid jump-cuts that switch topics abruptly. "
        "Return ONLY valid JSON (no markdown). "
        "Output schema: an array of objects {start: number, end: number, reason: string}. "
        "Rules: 0 <= start < end <= audio_duration. "
        "Return at most max_clips clips, in chronological order, with NO overlaps. "
        "Each clip length should be between min_clip_seconds and max_clip_seconds. "
        "Try to keep the total combined length <= target_total_seconds (never exceed 60 seconds). "
        "When possible, include a brief lead-in so the first sentence is not mid-thought. "
        "Choose moments that contain the key opinions, comparisons, and final takeaway."
    )

    user = {
        "audio_duration": round(dur, 2),
        "max_clips": int(max_clips),
        "target_total_seconds": float(target_total_seconds),
        "min_clip_seconds": float(min_clip_seconds),
        "max_clip_seconds": float(max_clip_seconds),
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

    clips: list[HighlightClip] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except Exception:
            continue
        reason = str(item.get("reason") or "").strip()
        clips.append(HighlightClip(start=start, end=end, reason=reason))

    cleaned = _clean_highlight_clips(
        clips,
        audio_duration=dur,
        max_clips=int(max_clips),
        min_clip_seconds=float(min_clip_seconds),
        max_clip_seconds=float(max_clip_seconds),
    )

    # Coherence guardrail: if the selected clips have large gaps, prefer a single contiguous excerpt.
    # Strategy: merge near-adjacent clips; if still disjoint, keep only the first block.
    merged: list[HighlightClip] = []
    max_gap_s = 2.25
    for c in cleaned:
        if not merged:
            merged.append(c)
            continue
        prev = merged[-1]
        gap = float(c.start) - float(prev.end)
        if gap <= max_gap_s:
            merged[-1] = HighlightClip(start=float(prev.start), end=float(c.end), reason=(prev.reason or c.reason))
        else:
            merged.append(c)

    if len(merged) > 1:
        # Keep only the earliest contiguous block for narrative consistency.
        merged = [merged[0]]

    cleaned = merged

    # Hard cap total length: keep <= target_total_seconds (and never exceed 60s).
    total = 0.0
    capped: list[HighlightClip] = []
    cap = min(60.0, max(10.0, float(target_total_seconds)))
    for c in cleaned:
        remaining = float(cap) - float(total)
        if remaining <= 0.0:
            break

        start = float(c.start)
        end = float(c.end)
        length = end - start
        if length <= 0.0:
            continue

        if length > remaining:
            # Truncate this clip to fit remaining budget.
            end = start + remaining
            length = end - start

        # If we can't fit at least the minimum meaningful clip length, stop.
        if length < max(0.05, float(min_clip_seconds) * 0.65):
            break

        capped.append(HighlightClip(start=start, end=end, reason=c.reason))
        total += length

    return capped


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
        "You are a video editor creating a slideshow plan. "
        "Your job: identify the MAIN TOPICS discussed and distribute slides across those topics. "
        "Return ONLY valid JSON (no markdown). "
        "The JSON must be an array of objects with keys: start, end, query. "
        "Optionally, objects may include: topic (string), topic_shift (boolean). "
        "Times are seconds; 0 <= start < end <= audio_duration. "
        "Use AT MOST max_images slides, but you may use fewer if the transcript has fewer major topic shifts. "
        "Hard rule: your slides MUST cover the full audio: first start=0 and the final end=audio_duration (within 0.2s). "
        "Queries must be short, concrete search keywords for finding real photos/stills suitable as background visuals. "
        "The query MUST reflect what is being discussed in that exact time window (characters/actors/scenes/setting/themes), not meta commentary. "
        "Avoid channel/creator names and avoid generic queries like 'best TV shows 2025' unless the transcript is explicitly about that. "
        "If the transcript is about a TV series, queries should include keywords like 'TV series still', 'cast', or 'scene still'. "
        "Hard rule: each query MUST include the topic name unless the segment clearly shifts topics; in that case set topic_shift=true and you may omit the topic. "
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
    for item in parsed:
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

    cleaned = cleaned[:max_images]
    # Ensure full coverage: stretch last slide to audio end.
    if cleaned:
        last = cleaned[-1]
        cleaned[-1] = StorySlide(start=last.start, end=float(audio_duration), query=last.query)

    return cleaned


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
        "Use AT MOST max_slides slides, but you may use fewer if the narration has fewer major topic shifts. "
        "Hard rule: your slides MUST cover the full audio: first start=0 and the final end=audio_duration (within 0.2s). "
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
            " This is for a YouTube Short with retention-focused visuals. "
            "Prefer MANY quick beats when max_slides allows (often 12-24 slides for a 30-45s short). "
            "Aim for ~1.5-2.2 seconds per slide on average when audio_duration <= 60. "
            "headline should be 2-4 words (keyword style), not a sentence; avoid filler; no spoilers. "
            "subhead should usually be omitted for Shorts unless essential. "
            "query should be biased toward: 'official poster', 'close up', 'scene still', 'actor face', 'cast still'."
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
    for item in parsed:
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

    cleaned = cleaned[:max_slides]
    if cleaned:
        last = cleaned[-1]
        cleaned[-1] = RichStorySlide(
            start=last.start,
            end=float(audio_duration),
            query=last.query,
            headline=last.headline,
            subhead=last.subhead,
        )

    return cleaned


def generate_video_title_with_llm(
    segments: list[TranscriptSegment],
    *,
    topic: str | None,
    channel_name: str,
    video_type: str = "review",
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

    vt = (video_type or "review").strip().lower()
    system = (
        "You are a YouTube video producer. "
        "Write ONE punchy, click-worthy title based on the provided transcript snippet. "
        "Return ONLY valid JSON (no markdown). Schema: {\"title\": <string>}. "
        "Hard rules: "
        "- Keep it under 70 characters. "
        "- Do NOT include the channel name. "
        "- Avoid profanity; keep it advertiser-safe. "
        "- No hashtags, no quotes around the whole title, no emojis. "
        "- It must reflect the transcript's actual stance/themes. "
        "- If this is NOT a review, do NOT include words like 'review' or 'verdict'. "
        "- Do NOT merely echo the topic_hint verbatim; rewrite into a human-friendly title." 
    )

    user = {
        "channel_name": channel_name,
        "topic_hint": topic or "",
        "video_type": vt,
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


def generate_video_title_fallback(audio_path: str | Path, *, topic: str | None, video_type: str = "review") -> str:
    vt = (video_type or "review").strip().lower()

    t = (topic or "").strip()
    if t:
        cleaned = (
            t.replace("(TV series)", "")
            .replace("TV series", "")
            .replace("tv series", "")
            .replace("_", " ")
            .replace("-", " ")
            .strip(" -|:")
        )
        # Fix common keyword clumps.
        cleaned_l = cleaned.lower()
        cleaned_l = cleaned_l.replace("conspiracytheories", "conspiracy theories")
        cleaned_l = cleaned_l.replace("toptheories", "top theories")
        cleaned = cleaned_l

        import re

        tokens = [x for x in re.split(r"[^a-z0-9]+", cleaned.lower()) if x]
        stem = Path(audio_path).stem.lower().replace("_", " ")

        # If we can detect a main proper topic word (like a show name), build a nicer title.
        # Keep this intentionally small + heuristic.
        show = None
        if "severance" in tokens or "severance" in stem:
            show = "Severance"

        has_theory = any(k in tokens for k in ("theory", "theories", "conspiracy")) or ("theor" in stem) or ("conspiracy" in stem)
        if vt != "review" and show and has_theory:
            return f"{show} Conspiracy Theories"

        # Otherwise, title-case the cleaned hint.
        cleaned_title = " ".join(w.capitalize() if w else "" for w in cleaned.split()).strip()
        if vt == "review":
            return f"{cleaned_title} Review".strip()

        if ("conspiracy" in stem) or ("theor" in stem) or has_theory:
            # Prefer "<Topic> Conspiracy Theories" structure for non-review.
            return f"{cleaned_title} Conspiracy Theories".strip()
        if ("explained" in stem) or ("explainer" in stem) or ("breakdown" in stem) or ("deep" in stem):
            return f"{cleaned_title} Explained".strip()
        return cleaned_title.strip() or cleaned_title

    p = Path(audio_path)
    stem = p.stem.replace("_", " ").strip()
    if vt == "review":
        return f"{stem[:80]} Review".strip() if stem else "Brutally Honest Review"
    return stem[:80] if stem else "Video"
