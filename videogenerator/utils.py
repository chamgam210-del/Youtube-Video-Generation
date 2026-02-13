from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Slide


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "you",
    "your",
}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return name[:180] if len(name) > 180 else name


def extract_keywords(text: str, max_words: int = 8) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z']+", text.lower())
    words = [w.strip("'") for w in words if w not in _STOPWORDS and len(w) >= 3]
    # keep order, unique
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= max_words:
            break
    return out


def first(iterable: Iterable[Any], default: Any = None) -> Any:
    for x in iterable:
        return x
    return default


def env_truthy(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def enforce_first_slide_seconds(
    slides: list["Slide"],
    *,
    first_s: float,
) -> list["Slide"]:
    """Force slide[0] duration to first_s seconds.

    Recomputes slide start/end times sequentially and keeps the final end time unchanged.
    If first_s is too large to fit, it is capped to leave at least 0.1s for each remaining slide.
    """

    if not slides:
        return slides

    try:
        from .models import Slide  # local import to avoid cycles
    except Exception:
        return slides

    ordered = sorted(slides, key=lambda s: float(s.start))
    if len(ordered) == 1:
        s0 = ordered[0]
        end = float(s0.start) + max(0.1, float(first_s))
        return [
            Slide(
                start=float(s0.start),
                end=end,
                image_path=s0.image_path,
                query=s0.query,
                headline=getattr(s0, "headline", None),
                subhead=getattr(s0, "subhead", None),
                source_page=s0.source_page,
                image_url=s0.image_url,
                license_name=s0.license_name,
                license_url=s0.license_url,
                attribution=s0.attribution,
                motion=getattr(s0, "motion", None),
            )
        ]

    total_end = float(ordered[-1].end)
    n = len(ordered)
    target_first = max(0.1, float(first_s))

    # Original per-slide durations.
    durs = [max(0.1, float(s.end) - float(s.start)) for s in ordered]
    # Cap target to fit remaining minimum durations.
    max_first = max(0.1, total_end - (0.1 * float(n - 1)))
    durs[0] = min(max_first, target_first)

    out: list[Slide] = []
    cur = 0.0
    # Rebuild all but last; last is stretched/compressed to hit total_end.
    for i in range(n - 1):
        dur = max(0.1, float(durs[i]))
        # Leave room for the remaining slides (at least 0.1s each).
        remaining_min = 0.1 * float((n - 1) - i)
        end = cur + dur
        if end > (total_end - remaining_min):
            end = max(cur + 0.1, total_end - remaining_min)

        s = ordered[i]
        out.append(
            Slide(
                start=float(cur),
                end=float(end),
                image_path=s.image_path,
                query=s.query,
                headline=getattr(s, "headline", None),
                subhead=getattr(s, "subhead", None),
                source_page=s.source_page,
                image_url=s.image_url,
                license_name=s.license_name,
                license_url=s.license_url,
                attribution=s.attribution,
                motion=getattr(s, "motion", None),
            )
        )
        cur = float(end)

    last = ordered[-1]
    last_start = cur
    last_end = max(last_start + 0.1, total_end)
    out.append(
        Slide(
            start=float(last_start),
            end=float(last_end),
            image_path=last.image_path,
            query=last.query,
            headline=getattr(last, "headline", None),
            subhead=getattr(last, "subhead", None),
            source_page=last.source_page,
            image_url=last.image_url,
            license_name=last.license_name,
            license_url=last.license_url,
            attribution=last.attribution,
            motion=getattr(last, "motion", None),
        )
    )

    return out
