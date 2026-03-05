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
_RED = _ass_color("#FF0000")
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


# ── Caption spelling fix from script ───────────────────────────────────────

def fix_caption_spelling(
    word_data: list[dict[str, Any]],
    script_sections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fix misspelled Whisper words using the ground-truth script as reference.

    This is a **text-only** correction — Whisper's timestamps and word count
    are never modified.  For each Whisper word, if a close match exists in
    the script vocabulary, the text is replaced.

    Parameters
    ----------
    word_data:
        Whisper output: ``[{"word": str, "start": float, "end": float}, ...]``.
    script_sections:
        Parsed script sections (needs ``"text"`` key per section).

    Returns
    -------
    corrected:
        Same length as *word_data*, same timestamps, just text fixes.
    corrections:
        Human-readable list of ``"whisper → script"`` changes.
    """
    import difflib
    import re

    if not word_data:
        return list(word_data), []

    # Build script vocabulary (unique lower-case → preferred casing).
    # Split on whitespace first, then further split on punctuation like
    # slashes, em-dashes, etc. so compound tokens ("assimilation/colonization")
    # become individual vocab entries.
    _split_re = re.compile(r"[/—–\-]+")
    vocab: dict[str, str] = {}
    for sec in script_sections:
        for raw_w in (sec.get("text") or "").split():
            parts = _split_re.split(raw_w)
            for p in parts:
                clean = p.strip(".,!?…;:'\"\u2018\u2019\u201c\u201d()")
                key = clean.lower()
                if key and len(key) >= 2 and key not in vocab:
                    vocab[key] = clean  # keep first occurrence's casing

    if not vocab:
        return list(word_data), []

    _PUNCT = ".,!?…;:'\"\u2018\u2019\u201c\u201d()"

    corrected: list[dict[str, Any]] = []
    corrections: list[str] = []

    for wd in word_data:
        original = str(wd.get("word", ""))
        stripped = original.strip()
        key = stripped.lower().strip(_PUNCT)

        if not key or key in vocab:
            # Exact match or empty — keep original word unchanged.
            corrected.append(dict(wd))
        elif len(key) < 5:
            # Short words (< 5 chars) are almost never misspelled by
            # Whisper in a meaningful way and cause too many false
            # positives (e.g. "four" → "for", "fan" → "Fans").
            corrected.append(dict(wd))
        else:
            # No exact match — find closest word in script vocab.
            # Only consider vocab entries of similar length (±2 chars).
            close = [
                v for v in vocab
                if abs(len(v) - len(key)) <= 2
            ]
            candidates = difflib.get_close_matches(key, close, n=1, cutoff=0.82)
            if candidates:
                best = candidates[0]
                # Preserve leading whitespace and trailing punctuation
                # from the original Whisper word.
                leading = original[: len(original) - len(stripped)]
                trail_start = len(stripped) - len(stripped.rstrip(_PUNCT))
                trailing = stripped[-trail_start:] if trail_start else ""
                new_word = leading + vocab[best] + trailing
                corrected.append({
                    "word": new_word,
                    "start": wd["start"],
                    "end": wd["end"],
                })
                corrections.append(f"{stripped} → {vocab[best]}")
            else:
                corrected.append(dict(wd))

    return corrected, corrections


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
    style: str = "word_highlight",   # word_highlight | pop
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
        ``"static"`` — show phrase text without per-word highlight animation.
    offset_seconds:
        Shift all word timestamps by this many seconds (e.g. to account
        for intro silence padding added before the narration).

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

    # Y-position: lower area for portrait to avoid blocking image center,
    # lower-third for landscape.
    if is_portrait:
        y_pos = int(height * 0.72)  # lower third — keeps key image content visible
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

        # Build clean word list for this phrase up-front.
        clean_words: list[str] = []
        for pw in phrase:
            c = _clean_word(pw.get("word", ""))
            if c:
                clean_words.append(c.upper())
        if not clean_words:
            continue

        if style == "static":
            # Static mode: show the whole phrase for its full duration,
            # no per-word colour highlight animation.
            phrase_text = " ".join(
                _clean_word(pw.get("word", "")).upper()
                for pw in phrase
                if _clean_word(pw.get("word", ""))
            )
            if not phrase_text.strip():
                continue
            prefix = f"{{\\an5\\pos({x_pos},{y_pos})}}"
            line = (
                f"Dialogue: 0,{_ass_time(phrase_start)},{_ass_time(phrase_end)},"
                f"Default,,0,0,0,,{prefix}{phrase_text}"
            )
            events.append(line)
        else:
            # Animated mode: one dialogue event per word with colour highlight.
            for wi, active_word in enumerate(phrase):
                w_start = float(active_word["start"]) + _off
                # End time: extend to the start of the NEXT word (or phrase end)
                # so there's no gap where no highlight is shown.
                if wi + 1 < len(phrase):
                    w_end = float(phrase[wi + 1]["start"]) + _off
                else:
                    w_end = phrase_end
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

    # ── Subscribe arrow CTA ──────────────────────────────────────────────
    # If the speaker says "subscribe" near the end, add an animated arrow
    # pointing down toward the YouTube Shorts subscribe button.
    is_portrait = height > width
    if is_portrait:
        sub_events = _subscribe_arrow_events(
            words, width, height, offset_seconds=_off,
        )
        events.extend(sub_events)

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


# ── Subscribe arrow CTA ─────────────────────────────────────────────────────

def _find_subscribe_word(
    words: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the last occurrence of 'subscribe' in the word list, or None."""
    result = None
    for w in words:
        cleaned = re.sub(r"[^a-zA-Z]", "", str(w.get("word", "")))
        if cleaned.lower() == "subscribe":
            result = w
    return result


def _subscribe_arrow_events(
    words: list[dict[str, Any]],
    width: int,
    height: int,
    *,
    offset_seconds: float = 0.0,
    duration: float = 4.0,
) -> list[str]:
    """Generate ASS dialogue lines for an animated subscribe arrow CTA.

    Triggered by the word "subscribe" in the transcript.  Shows:
    1. "SUBSCRIBE" text in a red pill-shaped banner (pop-in)
    2. Three bouncing arrows (▼) pointing down toward the YouTube subscribe button
    3. Everything fades out after *duration* seconds

    On YouTube Shorts (9:16), the subscribe button sits at the bottom-right
    area.  We place our arrow centered near the bottom to draw attention
    downward.
    """
    sub_word = _find_subscribe_word(words)
    if sub_word is None:
        return []

    _off = float(offset_seconds)
    t_start = float(sub_word["start"]) + _off
    t_end = t_start + duration

    # Colours in ASS &HAABBGGRR format
    red_fill = _ass_color("#FF0000")       # red
    dark_red = _ass_color("#990000")       # darker red for outline
    white = _ass_color("#FFFFFF")
    black_ol = _ass_color("#000000")

    cx = width // 2
    sub_y = int(height * 0.70)       # "SUBSCRIBE" banner position
    arrow_base_y = int(height * 0.78)  # first arrow position

    events: list[str] = []

    # ── 1. Red pill background (ASS vector drawing) ──
    pill_w, pill_h = 420, 90
    hw, hh = pill_w // 2, pill_h // 2
    r = hh  # corner radius
    pill_drawing = (
        f"m {-hw + r} {-hh} "
        f"b {-hw} {-hh} {-hw} {-hh} {-hw} {-hh + r} "
        f"l {-hw} {hh - r} "
        f"b {-hw} {hh} {-hw} {hh} {-hw + r} {hh} "
        f"l {hw - r} {hh} "
        f"b {hw} {hh} {hw} {hh} {hw} {hh - r} "
        f"l {hw} {-hh + r} "
        f"b {hw} {-hh} {hw} {-hh} {hw - r} {-hh} "
    )

    # Pop-in: 0% → 115% → 100% over 300ms
    pop_in = (
        "{\\fscx0\\fscy0"
        "\\t(0,150,\\fscx115\\fscy115)"
        "\\t(150,300,\\fscx100\\fscy100)}"
    )
    fade_tag = "{\\fad(0,500)}"

    # Red pill background
    events.append(
        f"Dialogue: 1,{_ass_time(t_start)},{_ass_time(t_end)},"
        f"Default,,0,0,0,,"
        f"{{\\an5\\pos({cx},{sub_y})"
        f"\\1c{red_fill}\\3c{dark_red}\\bord3\\shad0"
        f"\\p1}}{pop_in}{fade_tag}{pill_drawing}"
    )

    # "SUBSCRIBE" text on top of pill
    events.append(
        f"Dialogue: 2,{_ass_time(t_start)},{_ass_time(t_end)},"
        f"Default,,0,0,0,,"
        f"{{\\an5\\pos({cx},{sub_y})"
        f"\\fs52\\b1"
        f"\\1c{white}\\3c{dark_red}\\bord2\\shad0"
        f"\\fnArial Black}}"
        f"{pop_in}{fade_tag}SUBSCRIBE"
    )

    # ── 2. Three bouncing arrows (▼) ──
    # Each arrow pops in with a stagger, then pulses (scale bounce) in a
    # repeating cycle.  ASS \\t doesn't loop, so we emit short segments.
    arrow_spacing = 50
    bounce_period_ms = 500
    bounce_period_s = bounce_period_ms / 1000.0

    for i in range(3):
        a_y = arrow_base_y + i * arrow_spacing
        stagger_ms = i * 150  # each arrow appears 150ms after the previous

        arrow_start = t_start + stagger_ms / 1000.0

        # Number of bounce cycles that fit in the remaining duration
        remaining = t_end - arrow_start
        n_cycles = min(int(remaining / bounce_period_s) + 1, 15)

        for cyc in range(n_cycles):
            seg_start = arrow_start + cyc * bounce_period_s
            seg_end = min(seg_start + bounce_period_s, t_end)
            if seg_end <= seg_start + 0.05:
                break

            half = bounce_period_ms // 2
            # Pop-in only on the first cycle
            first_pop = ""
            if cyc == 0:
                first_pop = (
                    f"\\fscx0\\fscy0"
                    f"\\t(0,80,\\fscx120\\fscy120)"
                    f"\\t(80,160,\\fscx100\\fscy100)"
                )

            events.append(
                f"Dialogue: 2,{_ass_time(seg_start)},{_ass_time(seg_end)},"
                f"Default,,0,0,0,,"
                f"{{\\an5\\pos({cx},{a_y})"
                f"\\fs55\\b1"
                f"\\1c{red_fill}\\3c{black_ol}\\bord3\\shad0"
                f"\\fnArial"
                f"{first_pop}"
                f"\\t(0,{half},\\fscy115)"
                f"\\t({half},{bounce_period_ms},\\fscy100)"
                f"}}{fade_tag}▼"
            )

    return events


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


# ── On-screen text overlays (scripted shorts) ───────────────────────────────

_NORM_WORD_RE = re.compile(r"[^\w']", re.UNICODE)


def _normalize_for_match(w: str) -> str:
    """Lowercase + strip punctuation for fuzzy word matching."""
    return _NORM_WORD_RE.sub("", w.lower()).strip("'")


def _match_sections_to_audio(
    sections: list[dict],
    words: list[dict[str, Any]],
) -> list[float]:
    """Return the audio start-time for each section.

    Uses a greedy forward search: for each section (in script order),
    the first few narration words are fuzzy-matched against Whisper
    word timestamps, searching forward from the previous match position.
    """
    results: list[float] = []
    search_from = 0
    n_words = len(words)

    if not words:
        return [0.0] * len(sections)

    for sec in sections:
        # Build match text: label + on_screen_text + narration.
        # The user typically speaks the section label / number announcement
        # ("Number one, Vampires as colonizers") BEFORE the body narration,
        # so matching against the label gives an earlier, correct trigger.
        label = sec.get("label", "") or ""
        osd = sec.get("on_screen_text", "") or ""
        narration = sec.get("text", "") or ""
        # Strip numbering prefixes like "1)" from OSD — the user says
        # "number one" not "one parenthesis".
        import re as _re_inner
        osd_clean = _re_inner.sub(r'^\d+[)\.]\s*', '', osd).strip()
        match_text = f"{label} {osd_clean} {narration}".strip()
        tokens = [_normalize_for_match(w) for w in match_text.split()]
        match_tokens = [t for t in tokens if len(t) > 1][:6]

        if not match_tokens:
            t = float(words[min(search_from, n_words - 1)]["start"])
            results.append(t)
            continue

        best_idx = search_from
        best_score = -1
        # Accept threshold: once we find a window scoring at least this well,
        # stop scanning — prevents latching onto a later better-scoring window
        # which causes the image to appear after the narration has begun.
        _accept_thresh = max(len(match_tokens), len(match_tokens) * 2 - 1)

        for i in range(search_from, n_words):
            score = 0
            for j, mt in enumerate(match_tokens):
                if i + j >= n_words:
                    break
                wt = _normalize_for_match(words[i + j].get("word", ""))
                if mt == wt:
                    score += 2
                elif mt in wt or wt in mt:
                    score += 1
            if score > best_score:
                best_score = score
                best_idx = i
            # Stop as soon as we hit a good-enough match — don't scan ahead.
            if best_score >= _accept_thresh:
                break

        # Shift the slide start back slightly so the image appears just before
        # the matching word is spoken, not after.
        _pre_roll = 0.15
        results.append(max(0.0, float(words[best_idx]["start"]) - _pre_roll))
        search_from = best_idx + max(1, len(match_tokens) // 2)

    return results


def generate_on_screen_text_ass(
    script_sections: list[dict],
    words: list[dict[str, Any]],
    out_path: str | Path,
    *,
    width: int = 1080,
    height: int = 1920,
    offset_seconds: float = 0.0,
    display_duration: float = 3.5,
    existing_ass_path: str | Path | None = None,
) -> Path | None:
    """Generate audio-synced on-screen text overlays as ASS events.

    Each section in *script_sections* that has an ``on_screen_text`` value
    gets a timed overlay styled as an eye-catching title card with
    coloured text, positioned in the upper area of the frame.

    Timing is derived from Whisper word timestamps — the text appears
    when its section's narration is spoken, NOT at the script timestamps.

    If *existing_ass_path* is given, the events are appended to that file
    (allowing captions + OSD in a single ASS).  Otherwise a standalone
    ASS file is created.

    Returns the written file path, or ``None`` if there were no OSD items.
    """
    out_path = Path(out_path)

    # Filter sections that have on-screen text.
    osd_indices = [
        i for i, s in enumerate(script_sections)
        if (s.get("on_screen_text") or "").strip()
    ]
    if not osd_indices:
        # Nothing to overlay — keep existing file untouched.
        if existing_ass_path and Path(existing_ass_path).exists():
            if Path(existing_ass_path).resolve() != out_path.resolve():
                import shutil
                shutil.copy2(existing_ass_path, out_path)
            return out_path
        return None

    # Match ALL sections to audio for correct forward-search ordering.
    all_starts = _match_sections_to_audio(script_sections, words)

    is_portrait = height > width
    _off = float(offset_seconds)

    # ── Title-card styling ──
    font_size = max(52, int(height * 0.048))
    y_pos = int(height * 0.22) if is_portrait else int(height * 0.18)
    x_pos = width // 2
    max_chars_line = max(10, int(width * 0.78 / max(1, font_size * 0.52)))

    # Rotating accent colours (bright, eye-catching, never white).
    _ACCENT_COLORS = [
        "&H0042D4FF",  # gold/yellow  (#FFD442 → ASS BGR)
        "&H00FF6633",  # coral/orange (#3366FF → ASS BGR)
        "&H0000CCFF",  # bright cyan  (#FFCC00 → ASS BGR)
        "&H004040FF",  # red          (#FF4040 → ASS BGR)
        "&H0000FF80",  # green        (#80FF00 → ASS BGR)
        "&H00FFAA00",  # teal/blue    (#00AAFF → ASS BGR)
    ]

    events: list[str] = []

    for ci, sec_idx in enumerate(osd_indices):
        sec = script_sections[sec_idx]
        raw_text = (sec.get("on_screen_text") or "").strip()
        if not raw_text:
            continue

        t_start = all_starts[sec_idx] + _off

        # OSD stays visible for the ENTIRE slide / section duration.
        # End time = start of the NEXT section (any section, not just OSD ones).
        if sec_idx + 1 < len(all_starts):
            t_end = all_starts[sec_idx + 1] + _off - 0.15  # tiny gap before next slide
        else:
            # Last section — use display_duration as fallback.
            t_end = t_start + display_duration
        t_end = max(t_end, t_start + 1.0)  # at least 1 second

        # ── Word-wrap and prepare display text ──
        lines: list[str] = []
        for src_line in raw_text.split("\n"):
            src_line = src_line.strip()
            if not src_line:
                continue
            cur_words = src_line.split()
            cur = ""
            for cw in cur_words:
                test = f"{cur} {cw}".strip()
                if len(test) > max_chars_line and cur:
                    lines.append(cur.upper())
                    cur = cw
                else:
                    cur = test
            if cur:
                lines.append(cur.upper())

        if not lines:
            continue

        accent = _ACCENT_COLORS[ci % len(_ACCENT_COLORS)]

        # ── Animations ──
        # Slide-up entrance + fade out.
        slide_dist = int(font_size * 0.8)
        entrance = (
            f"\\move({x_pos},{y_pos + slide_dist},{x_pos},{y_pos},0,250)"
            f"\\fad(0,350)"
            f"\\t(0,200,\\frz0)"  # subtle settle
        )

        # Each line rendered separately with staggered entrance for title feel.
        line_h = int(font_size * 1.4)
        block_top = y_pos - (len(lines) * line_h) // 2

        # ── Layer 6 — semi-transparent dark background panel ──
        # Ensures OSD text is visible on ANY background image.
        total_block_h = len(lines) * line_h
        panel_cy = block_top + total_block_h // 2
        panel_cy_from = panel_cy + slide_dist
        max_chars = max(len(l) for l in lines)
        approx_text_w = int(max_chars * font_size * 0.52)
        panel_hw = min(
            approx_text_w // 2 + int(font_size * 0.7),
            width // 2 - 10,
        )
        panel_hh = total_block_h // 2 + int(font_size * 0.45)
        # Rounded-corner rectangle via cubic bezier arcs.
        cr = min(int(font_size * 0.25), panel_hw // 4, panel_hh // 4)
        draw_rect = (
            f"m {-panel_hw + cr} {-panel_hh} "
            f"l {panel_hw - cr} {-panel_hh} "
            f"b {panel_hw} {-panel_hh} {panel_hw} {-panel_hh} {panel_hw} {-panel_hh + cr} "
            f"l {panel_hw} {panel_hh - cr} "
            f"b {panel_hw} {panel_hh} {panel_hw} {panel_hh} {panel_hw - cr} {panel_hh} "
            f"l {-panel_hw + cr} {panel_hh} "
            f"b {-panel_hw} {panel_hh} {-panel_hw} {panel_hh} {-panel_hw} {panel_hh - cr} "
            f"l {-panel_hw} {-panel_hh + cr} "
            f"b {-panel_hw} {-panel_hh} {-panel_hw} {-panel_hh} {-panel_hw + cr} {-panel_hh}"
        )
        events.append(
            f"Dialogue: 6,{_ass_time(t_start)},{_ass_time(t_end)},"
            f"Default,,0,0,0,,"
            f"{{\\an5\\move({x_pos},{panel_cy_from},{x_pos},{panel_cy},0,280)"
            f"\\1c&H00000000\\1a&H66\\bord0\\shad0\\p1"
            f"\\fad(200,350)}}"
            f"{draw_rect}"
        )

        for li, line_text in enumerate(lines):
            line_y = block_top + li * line_h + line_h // 2
            stagger_ms = li * 80  # each line 80ms after previous
            line_start = t_start + stagger_ms / 1000.0

            # Slide-up per line.
            ly_from = line_y + slide_dist
            move_tag = f"\\move({x_pos},{ly_from},{x_pos},{line_y},0,280)"

            # Scale pop: 105% → 100%.
            pop = "\\fscx105\\fscy105\\t(0,250,\\fscx100\\fscy100)"

            # Layer 7 — drop shadow (offset black text).
            shadow_offset = max(2, font_size // 20)
            events.append(
                f"Dialogue: 7,{_ass_time(line_start)},{_ass_time(t_end)},"
                f"Default,,0,0,0,,"
                f"{{\\an5\\move({x_pos + shadow_offset},{ly_from + shadow_offset},"
                f"{x_pos + shadow_offset},{line_y + shadow_offset},0,280)"
                f"\\fs{font_size}\\b1\\fnArial Black\\fsp3"
                f"\\1c&H00000000\\3c&H00000000\\bord0\\shad0\\blur2"
                f"\\1a&H60"
                f"{pop}\\fad(0,350)}}{line_text}"
            )

            # Layer 8 — main coloured text with thick outline.
            events.append(
                f"Dialogue: 8,{_ass_time(line_start)},{_ass_time(t_end)},"
                f"Default,,0,0,0,,"
                f"{{\\an5{move_tag}"
                f"\\fs{font_size}\\b1\\fnArial Black\\fsp3"
                f"\\1c{accent}\\3c&H00000000\\bord7\\shad0\\blur0.5"
                f"{pop}\\fad(0,350)}}{line_text}"
            )

    if not events:
        return None

    # ── Write or merge into ASS file ──
    if existing_ass_path and Path(existing_ass_path).exists():
        content = Path(existing_ass_path).read_text(encoding="utf-8")
        content = content.rstrip("\n") + "\n" + "\n".join(events) + "\n"
    else:
        header = _ass_header(width, height, "Arial Black", font_size, 5, "#FFFFFF")
        content = header + "\n".join(events) + "\n"

    out_path.write_text(content, encoding="utf-8")
    return out_path


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
