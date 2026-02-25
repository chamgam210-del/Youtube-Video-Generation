"""Animated caption generation (ASS subtitles) — trending short-form styles.

Generates word-by-word highlighted captions in ASS (Advanced SubStation Alpha)
format, suitable for overlay via FFmpeg's ``ass`` filter.

Available styles (set via ``style`` parameter):

- ``"pop"`` *(default)* — **Hormozi / MrBeast bold**.  Large white text,
  active word pops to a vivid accent colour with a slight scale bump.
  2-3 word phrases for punchy readability.  Thick dark outline + drop shadow.
- ``"box_highlight"`` — **CapCut "Background Box"**.  Active word gets a
  coloured rounded-rectangle highlight behind it (via ``\\3c`` + ``BorderStyle=3``).
  Clean modern look used by most viral 2025-era shorts.
- ``"glow"`` — **Neon glow**.  White text with a soft coloured glow outline on
  the active word.  Eye-catching on dark/cinematic footage.
- ``"word_highlight"`` — Legacy plain colour-swap, no animation.

Usage::

    from videogenerator.captions import generate_ass_captions

    generate_ass_captions(
        words=word_list,         # list of {word, start, end}
        out_path="captions.ass",
        width=1080,
        height=1920,
        style="pop",             # or "box_highlight", "glow"
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

# Trendy accent colours for caption highlights.
_ACCENT_CYAN = _ass_color("#00F0FF")     # electric cyan
_ACCENT_GREEN = _ass_color("#39FF14")    # neon green
_ACCENT_PINK = _ass_color("#FF2D87")     # hot pink
_ACCENT_ORANGE = _ass_color("#FF6B00")   # vibrant orange


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
    max_phrase_words: int = 3,
    style: str = "pop",   # pop | box_highlight | glow | word_highlight
    offset_seconds: float = 0.0,
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
        Font family name visible to FFmpeg/libass.
    font_size:
        Override font size.  Default is auto-scaled per style.
    highlight_color:
        Hex ``#RRGGBB`` for the currently-spoken word.
    text_color:
        Hex ``#RRGGBB`` for non-active words.
    outline_px:
        Black outline thickness.  Default auto-scales with font size.
    max_phrase_words:
        Maximum words per visible phrase group.
    style:
        ``"pop"`` — Hormozi/MrBeast bold with scale pop + accent colour.
        ``"box_highlight"`` — CapCut-style coloured background box on active word.
        ``"glow"`` — Neon glow outline on active word.
        ``"word_highlight"`` — Legacy plain colour-swap (no animation).
    offset_seconds:
        Shift all word timestamps by this many seconds (e.g. to account
        for intro silence padding added before the narration).

    Returns
    -------
    Path to the written ``.ass`` file.
    """
    out_path = Path(out_path)
    is_portrait = height > width

    if not words:
        out_path.write_text(
            _ass_header(width, height, font_name, font_size or 72, 5, text_color, style=style),
            encoding="utf-8",
        )
        return out_path

    # ── Auto-scale defaults per style ────────────────────────────────
    if font_size is None:
        if style == "box_highlight":
            font_size = max(52, int(height * 0.046))   # slightly larger for box
        elif style == "glow":
            font_size = max(50, int(height * 0.044))
        else:
            font_size = max(52, int(height * 0.048))   # pop / word_highlight
    if outline_px is None:
        if style == "glow":
            outline_px = max(6, font_size // 8)        # thicker soft glow
        elif style == "box_highlight":
            outline_px = max(12, font_size // 4)       # box padding
        else:
            outline_px = max(4, font_size // 12)

    hi_col = _ass_color(highlight_color)
    txt_col = _ass_color(text_color)

    phrases = _group_words_into_phrases(words, max_words=max_phrase_words)

    # Y-position: centred for portrait, lower-third for landscape.
    if is_portrait:
        y_pos = int(height * 0.50)
    else:
        y_pos = int(height * 0.82)
    x_pos = width // 2

    events: list[str] = []
    _off = float(offset_seconds)

    for phrase in phrases:
        phrase_start = float(phrase[0]["start"]) + _off
        phrase_end = float(phrase[-1]["end"]) + _off
        if phrase_end <= phrase_start:
            continue

        clean_words: list[str] = []
        for pw in phrase:
            c = _clean_word(pw.get("word", ""))
            if c:
                clean_words.append(c.upper())
        if not clean_words:
            continue

        for wi, active_word in enumerate(phrase):
            w_start = float(active_word["start"]) + _off
            if wi + 1 < len(phrase):
                w_end = float(phrase[wi + 1]["start"]) + _off
            else:
                w_end = phrase_end
            if w_end <= w_start:
                w_end = w_start + 0.15

            # ── Build styled text per style ──────────────────────────
            if style == "box_highlight":
                text, prefix = _render_box_highlight(
                    phrase, wi, x_pos, y_pos, hi_col, txt_col, font_size,
                )
            elif style == "glow":
                text, prefix = _render_glow(
                    phrase, wi, x_pos, y_pos, hi_col, txt_col,
                )
            elif style == "pop":
                text, prefix = _render_pop(
                    phrase, wi, x_pos, y_pos, hi_col, txt_col, font_size,
                )
            else:  # word_highlight (legacy)
                text, prefix = _render_word_highlight(
                    phrase, wi, x_pos, y_pos, hi_col, txt_col,
                )

            if not text.strip():
                continue

            line = (
                f"Dialogue: 0,{_ass_time(w_start)},{_ass_time(w_end)},"
                f"Default,,0,0,0,,{prefix}{text}"
            )
            events.append(line)

    header = _ass_header(width, height, font_name, font_size, outline_px, text_color, style=style)

    out_path.write_text(
        header + "\n".join(events) + "\n",
        encoding="utf-8",
    )
    return out_path


# ── Style renderers ─────────────────────────────────────────────────────────

def _render_pop(
    phrase: list[dict], wi: int, x: int, y: int,
    hi_col: str, txt_col: str, font_size: int,
) -> tuple[str, str]:
    """Hormozi/MrBeast bold style — scale bump + accent colour on active word.

    - Active word: bright accent, scale 115% → 100% over 100ms, slight Y lift
    - Other words: white with heavy dark outline
    - Phrase entrance: subtle scale 108% → 100%
    """
    parts: list[str] = []
    for j, pw in enumerate(phrase):
        clean = _clean_word(pw.get("word", ""))
        if not clean:
            continue
        word_upper = clean.upper()
        if j == wi:
            # Active word: accent colour + scale pop
            parts.append(
                f"{{\\c{hi_col}\\fscx115\\fscy115"
                f"\\t(0,100,\\fscx100\\fscy100)}}"
                f"{word_upper}{{\\c{txt_col}\\fscx100\\fscy100}}"
            )
        else:
            parts.append(word_upper)

    text = " ".join(parts)
    # Phrase entrance pop on first word
    prefix = f"{{\\an5\\pos({x},{y})}}"
    if wi == 0:
        prefix += "{\\fscx108\\fscy108\\t(0,120,\\fscx100\\fscy100)}"
    return text, prefix


def _render_box_highlight(
    phrase: list[dict], wi: int, x: int, y: int,
    hi_col: str, txt_col: str, font_size: int,
) -> tuple[str, str]:
    """CapCut "Background Box" style — coloured box behind active word.

    Uses BorderStyle=3 (opaque box) via inline override on the active word,
    with a contrasting text colour for readability.
    """
    parts: list[str] = []
    for j, pw in enumerate(phrase):
        clean = _clean_word(pw.get("word", ""))
        if not clean:
            continue
        word_upper = clean.upper()
        if j == wi:
            # Active word: dark text on coloured box background.
            # \\3c sets outline/box colour, \\bord sets box padding,
            # \\c sets text colour (dark for contrast).
            dark_text = _ass_color("#000000")
            parts.append(
                f"{{\\c{dark_text}\\3c{hi_col}\\4c{hi_col}"
                f"\\bord{max(10, font_size // 5)}\\shad0"
                f"\\fscx105\\fscy105\\t(0,80,\\fscx100\\fscy100)}}"
                f"{word_upper}"
                f"{{\\c{txt_col}\\3c{_OUTLINE}\\4c{_SHADOW}"
                f"\\bord{max(3, font_size // 16)}\\shad2\\fscx100\\fscy100}}"
            )
        else:
            parts.append(word_upper)

    text = " ".join(parts)
    prefix = f"{{\\an5\\pos({x},{y})}}"
    if wi == 0:
        prefix += "{\\fscx106\\fscy106\\t(0,100,\\fscx100\\fscy100)}"
    return text, prefix


def _render_glow(
    phrase: list[dict], wi: int, x: int, y: int,
    hi_col: str, txt_col: str,
) -> tuple[str, str]:
    """Neon glow style — soft coloured outline glow on active word.

    Active word gets a thick soft-edged coloured outline (glow effect)
    plus a brighter text fill.
    """
    parts: list[str] = []
    for j, pw in enumerate(phrase):
        clean = _clean_word(pw.get("word", ""))
        if not clean:
            continue
        word_upper = clean.upper()
        if j == wi:
            # Active word: coloured outline "glow" + white text + slight scale
            parts.append(
                f"{{\\c{_WHITE}\\3c{hi_col}\\bord8\\blur3"
                f"\\fscx110\\fscy110\\t(0,100,\\fscx100\\fscy100)}}"
                f"{word_upper}"
                f"{{\\c{txt_col}\\3c{_OUTLINE}\\bord4\\blur0"
                f"\\fscx100\\fscy100}}"
            )
        else:
            parts.append(word_upper)

    text = " ".join(parts)
    prefix = f"{{\\an5\\pos({x},{y})}}"
    if wi == 0:
        prefix += "{\\fscx106\\fscy106\\t(0,120,\\fscx100\\fscy100)}"
    return text, prefix


def _render_word_highlight(
    phrase: list[dict], wi: int, x: int, y: int,
    hi_col: str, txt_col: str,
) -> tuple[str, str]:
    """Legacy plain colour-swap style — no animation."""
    parts: list[str] = []
    for j, pw in enumerate(phrase):
        clean = _clean_word(pw.get("word", ""))
        if not clean:
            continue
        word_upper = clean.upper()
        if j == wi:
            parts.append(f"{{\\c{hi_col}}}{word_upper}{{\\c{txt_col}}}")
        else:
            parts.append(word_upper)
    text = " ".join(parts)
    prefix = f"{{\\an5\\pos({x},{y})}}"
    return text, prefix


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
    *,
    style: str = "pop",
) -> str:
    primary = _ass_color(text_color)
    secondary = _ass_color("#FFFF00")
    outline_col = _OUTLINE
    shadow_col = _SHADOW

    # Style-specific tweaks.
    bold = -1          # -1 = bold
    spacing = 3        # wider letter spacing for readability
    shadow_depth = 3   # visible drop shadow for depth
    border_style = 1   # 1 = outline + shadow

    if style == "box_highlight":
        border_style = 3   # 3 = opaque background box
        shadow_depth = 0
        spacing = 4
    elif style == "glow":
        shadow_depth = 0
        spacing = 2

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
        f"{outline_col},{shadow_col},{bold},0,0,0,100,100,{spacing},0,{border_style},"
        f"{outline_px},{shadow_depth},5,20,20,30\n"
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
