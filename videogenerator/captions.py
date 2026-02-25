"""Animated caption generation (ASS subtitles) — CapCut / Hormozi style.

Generates word-by-word highlighted captions in ASS (Advanced SubStation Alpha)
format, suitable for overlay via FFmpeg's ``ass`` filter.

Style:
- Large bold text, centered slightly above middle (portrait) or lower-third (landscape)
- Current word highlighted in yellow; other words white with thick black outline
- Short phrases (3-5 words) shown at a time
- Subtle pop-in scale animation on each phrase

Usage::

    from videogenerator.captions import generate_ass_captions

    generate_ass_captions(
        words=word_list,         # list of {word, start, end}
        out_path="captions.ass",
        width=1080,
        height=1920,
    )
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# ── Colour helpers (ASS uses &HAABBGGRR) ──────────────────────────────────────

def _ass_color(hex_rgb: str, alpha: int = 0) -> str:
    """Convert ``#RRGGBB`` to ASS ``&HAABBGGRR``."""
    hex_rgb = hex_rgb.lstrip("#")
    r, g, b = int(hex_rgb[0:2], 16), int(hex_rgb[2:4], 16), int(hex_rgb[4:6], 16)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


_WHITE = _ass_color("#FFFFFF")
_YELLOW = _ass_color("#FFFF00")
_OUTLINE = _ass_color("#000000")
_SHADOW = _ass_color("#000000", alpha=0x60)


# ── Phrase grouping ──────────────────────────────────────────────────────────

def _group_words_into_phrases(
    words: list[dict[str, Any]],
    *,
    max_words: int = 4,
    max_gap: float = 0.7,
) -> list[list[dict[str, Any]]]:
    """Group word-level timestamps into short display phrases.

    Rules:
    - At most *max_words* per phrase.
    - A gap > *max_gap* seconds between consecutive words forces a new phrase.
    - Sentence-ending punctuation (``.?!``) ends the current phrase.
    """
    phrases: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for w in words:
        if current:
            gap = float(w["start"]) - float(current[-1]["end"])
            prev_text = str(current[-1].get("word", "")).strip()
            ends_sentence = bool(prev_text and prev_text[-1] in ".?!…")
            if len(current) >= max_words or gap > max_gap or ends_sentence:
                phrases.append(current)
                current = []
        current.append(w)

    if current:
        phrases.append(current)

    return phrases


# ── Timestamp formatting ────────────────────────────────────────────────────

def _ass_time(seconds: float) -> str:
    """Format seconds as ASS timestamp ``H:MM:SS.cc`` (centiseconds)."""
    s = max(0.0, float(seconds))
    # Convert to integer centiseconds to avoid rounding overflow.
    total_cs = int(round(s * 100))
    h = total_cs // 360000
    total_cs %= 360000
    m = total_cs // 6000
    total_cs %= 6000
    sec = total_cs // 100
    cs = total_cs % 100
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"


# ── ASS generation ──────────────────────────────────────────────────────────

def generate_ass_captions(
    words: list[dict[str, Any]],
    out_path: str | Path,
    *,
    width: int = 1080,
    height: int = 1920,
    font_name: str = "Arial Black",
    font_size: int | None = None,
    highlight_color: str = "#FFFF00",
    text_color: str = "#FFFFFF",
    outline_px: int | None = None,
    max_phrase_words: int = 4,
    style: str = "word_highlight",   # word_highlight | pop
) -> Path:
    """Generate an ASS subtitle file with animated word-by-word captions.

    Parameters
    ----------
    words:
        List of dicts with keys ``word`` (str), ``start`` (float), ``end`` (float).
    out_path:
        Destination ``.ass`` file path.
    width, height:
        Video resolution (used for ASS PlayRes and positioning).
    font_name:
        Font family name visible to FFmpeg/libass.  ``Arial Black`` is safe on all platforms.
    font_size:
        Override font size.  Default is auto-scaled to ~6.5 % of height.
    highlight_color:
        Hex ``#RRGGBB`` for the currently-spoken word.
    text_color:
        Hex ``#RRGGBB`` for non-active words.
    outline_px:
        Black outline thickness.  Default auto-scales with font size.
    max_phrase_words:
        Maximum words per visible phrase group.
    style:
        ``"word_highlight"`` — CapCut-style per-word colour highlight.
        ``"pop"`` — same, plus a subtle scale pop on each phrase.

    Returns
    -------
    Path to the written ``.ass`` file.
    """
    out_path = Path(out_path)

    if not words:
        # Write a minimal valid ASS to avoid FFmpeg errors.
        out_path.write_text(_ass_header(width, height, font_name, font_size or 72, 5, text_color), encoding="utf-8")
        return out_path

    # Auto-scale defaults.
    is_portrait = height > width
    if font_size is None:
        font_size = max(48, int(height * 0.043))  # ~82 for 1920
    if outline_px is None:
        outline_px = max(3, font_size // 16)

    hi_col = _ass_color(highlight_color)
    txt_col = _ass_color(text_color)

    phrases = _group_words_into_phrases(words, max_words=max_phrase_words)

    # Y-position: slightly above centre for portrait, lower-third for landscape.
    if is_portrait:
        y_pos = int(height * 0.50)  # dead centre
    else:
        y_pos = int(height * 0.82)

    x_pos = width // 2

    events: list[str] = []

    for phrase in phrases:
        phrase_start = float(phrase[0]["start"])
        phrase_end = float(phrase[-1]["end"])
        if phrase_end <= phrase_start:
            continue

        for wi, active_word in enumerate(phrase):
            w_start = float(active_word["start"])
            w_end = float(active_word["end"])
            if w_end <= w_start:
                w_end = w_start + 0.15

            # Build text with override tags: highlight the active word.
            parts: list[str] = []
            for j, pw in enumerate(phrase):
                clean = _clean_word(pw.get("word", ""))
                if not clean:
                    continue
                if j == wi:
                    parts.append(f"{{\\c{hi_col}}}{clean.upper()}{{\\c{txt_col}}}")
                else:
                    parts.append(clean.upper())

            text = " ".join(parts)
            if not text.strip():
                continue

            # Position + optional pop-in animation.
            prefix = f"{{\\an5\\pos({x_pos},{y_pos})}}"
            if style == "pop" and wi == 0:
                # Phrase entrance: scale 110% → 100% over 120 ms.
                prefix += "{\\fscx110\\fscy110\\t(0,120,\\fscx100\\fscy100)}"

            line = (
                f"Dialogue: 0,{_ass_time(w_start)},{_ass_time(w_end)},"
                f"Default,,0,0,0,,{prefix}{text}"
            )
            events.append(line)

    header = _ass_header(width, height, font_name, font_size, outline_px, text_color)

    out_path.write_text(
        header + "\n".join(events) + "\n",
        encoding="utf-8",
    )
    return out_path


# ── Helpers ──────────────────────────────────────────────────────────────────

_STRIP_RE = re.compile(r"[^\w''.,!?\-…]", re.UNICODE)


def _clean_word(w: str) -> str:
    """Strip Whisper artefacts (leading spaces, odd unicode) from a word."""
    return (w or "").strip()


def _ass_header(
    width: int,
    height: int,
    font_name: str,
    font_size: int,
    outline_px: int,
    text_color: str,
) -> str:
    primary = _ass_color(text_color)
    secondary = _ass_color("#FFFF00")
    outline_col = _OUTLINE
    shadow_col = _SHADOW

    return (
        "[Script Info]\n"
        "Title: Auto Captions\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV\n"
        f"Style: Default,{font_name},{font_size},{primary},{secondary},"
        f"{outline_col},{shadow_col},-1,0,0,0,100,100,2,0,1,"
        f"{outline_px},0,5,20,20,30\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


# ── Word-level timestamp extraction from Whisper cache ───────────────────────

def extract_words_from_whisper_cache(cache_path: str | Path) -> list[dict[str, Any]]:
    """Read word-level timestamps from a Whisper transcript cache JSON.

    Returns list of ``{"word": str, "start": float, "end": float}``.
    Falls back to empty list if no word data is present.
    """
    import json

    p = Path(cache_path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []

    words = data.get("words")
    if isinstance(words, list) and words:
        return words

    # Fallback: try to extract from segments[*].words
    out: list[dict[str, Any]] = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            if isinstance(w, dict) and "word" in w and "start" in w and "end" in w:
                out.append({"word": w["word"], "start": float(w["start"]), "end": float(w["end"])})
    return out


def words_from_segments_fallback(
    segments: list[dict[str, Any]] | list[Any],
) -> list[dict[str, Any]]:
    """Create approximate word timestamps by splitting segment text evenly.

    Used when word-level data is unavailable (e.g. old Whisper cache without
    ``word_timestamps=True``).
    """
    out: list[dict[str, Any]] = []
    for seg in segments:
        if isinstance(seg, dict):
            start, end, text = float(seg["start"]), float(seg["end"]), str(seg.get("text", ""))
        else:
            start, end, text = float(seg.start), float(seg.end), str(seg.text)
        words = text.strip().split()
        if not words:
            continue
        dur = max(0.01, end - start)
        per_word = dur / len(words)
        for i, w in enumerate(words):
            ws = start + i * per_word
            we = ws + per_word
            out.append({"word": w, "start": round(ws, 3), "end": round(we, 3)})
    return out
