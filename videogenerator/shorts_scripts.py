from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .llm_storyboard import _openai_chat_completions


@dataclass(frozen=True)
class ShortScriptLine:
    start: float
    end: float
    text: str
    keywords: list[str] | None = None
    image_query: str | None = None


@dataclass(frozen=True)
class ShortScript:
    title: str
    thumbnail_text: str
    lines: list[ShortScriptLine]


def format_script_bracketed(script: ShortScript) -> str:
    out: list[str] = []
    for ln in script.lines:
        out.append(f"[ {ln.start:.2f} - {ln.end:.2f}] {ln.text.strip()}")
    return "\n".join(out).strip() + "\n"


def _coerce_float(x: Any, *, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def generate_shorts_scripts_from_review_text_with_llm(
    review_text: str,
    *,
    topic: str | None = None,
    count: int = 5,
    target_total_seconds: float = 38.0,
    model: str = "gpt-5.2",
) -> list[ShortScript]:
    """Generate multiple retention-optimized Shorts scripts from raw review thoughts.

    Output is intended to be read verbatim for voiceover, producing 30–40s audio.
    """

    txt = " ".join((review_text or "").split()).strip()
    if not txt:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    n = int(count)
    if n <= 0:
        n = 1
    n = min(10, max(1, n))

    target = float(target_total_seconds)
    target = min(40.0, max(28.0, target))

    system = (
        "You are an expert YouTube Shorts scriptwriter and retention editor. "
        "Create high-retention SHORTS scripts from the user's raw review thoughts. "
        "Each short MUST follow this structure: HOOK (0-3s) -> TENSION/CONTROVERSY (3-10s) -> "
        "PROOF/EXAMPLES (10-25s) -> PAYOFF/STRONG TAKE (25-35s) -> LOOPABLE END (35-40s). "
        "Hard rules: delay the verdict until the PAYOFF section; do not open with generic phrases like 'my review is...'; "
        "end with a question; keep it non-spoilery; keep sentences short and speakable. "
        "Return ONLY valid JSON (no markdown). "
        "Schema: {\"shorts\": [ {\"title\": str, \"thumbnail_text\": str, \"lines\": ["
        "{\"start\": number, \"end\": number, \"text\": str, \"keywords\": [str], \"image_query\": str} ] } ] }. "
        "Timing rules: Each short starts at 0.0 and ends between 30 and 40 seconds. "
        "Lines must be contiguous (no gaps > 0.4s) and each line should be 2-7 seconds. "
        "thumbnail_text must be 2-4 words (clickable hook). "
        "keywords must be 1-3 words max (for big on-screen emphasis). "
        "image_query should be short, and biased toward: official poster, close-up still, actor face, dramatic still."
    )

    user = {
        "topic": (topic or "").strip(),
        "count": n,
        "target_total_seconds": round(target, 2),
        "review_text": txt[:8000],
    }

    content = _openai_chat_completions(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        timeout_s=90,
    )

    try:
        parsed = json.loads(content)
    except Exception as e:
        raise RuntimeError(f"LLM did not return valid JSON. Got: {content[:400]}") from e

    shorts_raw = (parsed or {}).get("shorts") if isinstance(parsed, dict) else None
    if not isinstance(shorts_raw, list):
        raise RuntimeError("LLM JSON must be an object with key 'shorts' (array)")

    out: list[ShortScript] = []
    for sh in shorts_raw[:n]:
        if not isinstance(sh, dict):
            continue
        title = str(sh.get("title") or "").strip() or "Short"
        thumb = str(sh.get("thumbnail_text") or "").strip() or title
        lines_raw = sh.get("lines")
        if not isinstance(lines_raw, list) or not lines_raw:
            continue

        lines: list[ShortScriptLine] = []
        for ln in lines_raw:
            if not isinstance(ln, dict):
                continue
            start = _coerce_float(ln.get("start"), default=0.0)
            end = _coerce_float(ln.get("end"), default=start)
            text = str(ln.get("text") or "").strip()
            if not text:
                continue
            if end <= start:
                continue
            kws = ln.get("keywords")
            if isinstance(kws, list):
                keywords = [str(k).strip() for k in kws if str(k).strip()][:5]
            else:
                keywords = None
            iq = str(ln.get("image_query") or "").strip() or None
            lines.append(ShortScriptLine(start=float(start), end=float(end), text=text, keywords=keywords, image_query=iq))

        if not lines:
            continue

        lines.sort(key=lambda x: x.start)

        # Minimal sanity: enforce starts at 0-ish and clamp total length.
        if lines[0].start > 0.75:
            shift = lines[0].start
            lines = [
                ShortScriptLine(
                    start=max(0.0, ln.start - shift),
                    end=max(0.0, ln.end - shift),
                    text=ln.text,
                    keywords=ln.keywords,
                    image_query=ln.image_query,
                )
                for ln in lines
            ]

        # Clamp end to 40s.
        clamped: list[ShortScriptLine] = []
        for ln in lines:
            s = max(0.0, float(ln.start))
            e = min(40.0, max(s, float(ln.end)))
            if e <= s:
                continue
            clamped.append(ShortScriptLine(start=s, end=e, text=ln.text, keywords=ln.keywords, image_query=ln.image_query))
        if not clamped:
            continue

        out.append(ShortScript(title=title, thumbnail_text=thumb, lines=clamped))

    if not out:
        raise RuntimeError("LLM returned no usable shorts scripts")

    return out
