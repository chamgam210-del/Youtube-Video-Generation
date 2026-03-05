"""Parse a timestamped script into sections for *Scripted Short* mode.

Supports **any** script format — the heavy lifting is done by an LLM
(``gpt-4o-mini``) that converts free-form scripts into structured JSON.

Examples of formats that work::

    Format A (em-dash)::

        0:00–0:03 — Hook (curiosity + promise)
        "Before A24 made it a movie… the Backrooms was just one photo."
        On-screen text: No location • No story

    Format B (bracketed)::

        [0:00–0:04] HOOK
        VO: "CBS just dropped a new show."
        ON-SCREEN TEXT: "MARSHALS (2026)"
        ON-SCREEN TEXT (small): "CBS • Paramount+"

    Format C (anything else with timestamps)::

        0:00-0:05 The intro
        Narrator says something here.
        Text overlay: Hello world
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field


@dataclass
class ScriptSection:
    """One section of a timestamped script."""

    start: float  # seconds
    end: float  # seconds
    label: str  # e.g. "Hook", "The photo"
    text: str  # narration text (quotes stripped)
    on_screen_text: str = ""  # e.g. "No location • No story"
    search_query: str = ""  # filled later by the pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_to_seconds(m: int, s: int) -> float:
    return float(m) * 60.0 + float(s)


def _strip_quotes(text: str) -> str:
    """Strip surrounding ASCII or smart quotes from *text*."""
    t = text.strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith('\u201c') and t.endswith('\u201d')):
        t = t[1:-1].strip()
    return t


def _parse_ts(ts: str) -> float:
    """Convert a timestamp string like ``1:30`` or ``01:30`` to seconds."""
    ts = ts.strip()
    parts = re.split(r"[:.]", ts)
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0.0


# ---------------------------------------------------------------------------
# LLM-based parser (primary)
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """\
You are a script parser. The user will give you a timestamped video script \
in ANY format. Your job is to extract EVERY section into structured JSON.

Return ONLY valid JSON (no markdown fences, no explanation):
{
  "sections": [
    {
      "start": "M:SS",
      "end": "M:SS",
      "label": "section title/label",
      "narration": "the spoken text (VO/narrator lines, stripped of quotes)",
      "on_screen_text": "overlay text if any, or empty string"
    }
  ]
}

RULES:
1. Each section has a timestamp range. Extract start and end as "M:SS".
2. The label is the section name/title (e.g. "HOOK", "The image", "CLOSE").
3. Narration is what the narrator/VO says — strip VO:/NARRATOR: prefixes \
   and surrounding quotes.
4. on_screen_text is text meant to appear on screen — look for lines like \
   "ON-SCREEN TEXT:", "On-screen:", "Text overlay:", "TITLE:", etc. \
   If a section has multiple on-screen text lines, join them with newlines. \
   Strip surrounding quotes from the values.
5. "Beat." lines or stage directions are NOT narration — skip them.
6. Preserve the EXACT order of sections.
7. Do NOT invent or modify content — extract verbatim from the script.
8. CRITICAL: If the script begins with a standalone intro/preamble sentence or \
   paragraph BEFORE the first numbered/titled section (e.g. "Here are some \
   interesting theories...", "Let me break down...", "Welcome back..."), \
   extract it as its own section with label "INTRO", on_screen_text "", \
   and use start "0:00". Give it an estimated end time based on text length \
   (~3 seconds per short sentence). The INTRO narration must NOT be merged \
   into the first real section — keep them separate.
9. The on_screen_text for a section should NOT include the section's narration \
   text — only explicit overlay/title instructions from the script author.
"""


def _parse_with_llm(raw_text: str) -> list[ScriptSection] | None:
    """Try to parse the script using the LLM.  Returns ``None`` on failure."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from .llm_storyboard import _openai_chat_completions
    except Exception:
        return None

    try:
        content = _openai_chat_completions(
            api_key=api_key,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            timeout_s=30,
        )

        # Strip markdown fences if the model added them.
        stripped = content.strip()
        fence = re.search(r"```(?:json)?\s*\n?(.*?)```", stripped, re.DOTALL)
        if fence:
            stripped = fence.group(1).strip()

        data = json.loads(stripped)
        raw_sections = data.get("sections", []) if isinstance(data, dict) else []
        if not raw_sections:
            return None

        sections: list[ScriptSection] = []
        for s in raw_sections:
            start = _parse_ts(str(s.get("start", "0:00")))
            end = _parse_ts(str(s.get("end", "0:00")))
            label = str(s.get("label", "")).strip()
            narration = _strip_quotes(str(s.get("narration", "")).strip())
            on_screen = _strip_quotes(str(s.get("on_screen_text", "")).strip())
            if end <= start:
                end = start + 5.0  # safety fallback
            sections.append(ScriptSection(
                start=start,
                end=end,
                label=label,
                text=narration,
                on_screen_text=on_screen,
            ))

        if sections:
            print(f"[script_parser] LLM parsed {len(sections)} sections successfully")
            return sections
    except Exception as e:
        print(f"[script_parser] LLM parsing failed ({e}), falling back to regex")

    return None


# ---------------------------------------------------------------------------
# Regex-based parser (fallback)
# ---------------------------------------------------------------------------

# Matches many timestamp header formats:
#   0:00–0:04 (Hook)                        — plain with parens
#   0:00–0:03 — Hook (curiosity + promise)  — em-dash + parens
#   [0:00–0:04] HOOK                        — bracketed
#   [0:00–0:04] HOOK (qualifier)            — bracketed + parens
_TS_RE = re.compile(
    r"^\s*"
    r"(?:\[\s*)?"                                          # optional opening bracket
    r"(?P<m1>\d{1,2}):(?P<s1>\d{2})"
    r"\s*[–\-—]+\s*"
    r"(?P<m2>\d{1,2}):(?P<s2>\d{2})"
    r"(?:\s*\])?"                                          # optional closing bracket
    r"(?:\s*[–\-—]+\s*(?P<dash_label>[^(\n]*?))?"
    r"(?:\s+(?P<bare_label>[A-Z][A-Z0-9 ']*[A-Z0-9]))?"  # ALL-CAPS label after bracket
    r"(?:\s*\((?P<paren_label>[^)]*)\))?"
    r"\s*$",
)

# Core alternation for OSD directive keywords.
_OSD_KW = (
    r"(?:"
    r"on[- ]?screen(?:\s+text)?"
    r"|text\s+on\s+screen"
    r"|visual(?:\s+text)?"
    r"|title\s*card"
    r"|screen\s+text"
    r"|overlay(?:\s+text)?"
    r")"
)

_ONSCREEN_RE = re.compile(
    r"^\s*" + _OSD_KW +
    r"(?:\s*\([^)]*\))?"          # optional qualifier — (small), (big), etc.
    r"\s*[:=]\s*(?P<text>.+)$",
    re.IGNORECASE,
)

_ONSCREEN_BRACKET_RE = re.compile(
    r"^\s*[\[\(]\s*"
    r"(?:on[- ]?screen(?:\s+text)?|visual(?:\s+text)?|title(?:\s+card)?|overlay(?:\s+text)?)"
    r"(?:\s*\([^)]*\))?"
    r"\s*[\]\)]\s*[:=]?\s*(?P<text>.+)$",
    re.IGNORECASE,
)

_ONSCREEN_BARE_RE = re.compile(
    r"^\s*" + _OSD_KW +
    r"(?:\s*\([^)]*\))?"
    r"\s*[:=]\s*$",
    re.IGNORECASE,
)

_SKIP_RE = re.compile(r"^\s*beat\.?\s*$", re.IGNORECASE)

_VO_PREFIX_RE = re.compile(
    r"^\s*(?:VO|V\.?O\.?|NARRATOR|VOICEOVER|VOICE[- ]?OVER|NARRATION)\s*:\s*",
    re.IGNORECASE,
)


def _parse_with_regex(raw_text: str) -> list[ScriptSection]:
    """Regex-based fallback parser."""

    lines = raw_text.splitlines()
    sections: list[ScriptSection] = []
    current_start: float | None = None
    current_end: float | None = None
    current_label: str = ""
    narration_buf: list[str] = []
    onscreen_buf: list[str] = []
    collecting_osd: bool = False

    def _flush() -> None:
        nonlocal current_start, current_end, current_label, narration_buf, onscreen_buf, collecting_osd
        if current_start is not None and current_end is not None:
            text = _strip_quotes(" ".join(narration_buf).strip())
            on_screen = "\n".join(onscreen_buf).strip()
            sections.append(
                ScriptSection(
                    start=current_start,
                    end=current_end,
                    label=current_label,
                    text=text,
                    on_screen_text=on_screen,
                )
            )
        current_start = None
        current_end = None
        current_label = ""
        narration_buf = []
        onscreen_buf = []
        collecting_osd = False

    for line in lines:
        m = _TS_RE.match(line)
        if m:
            _flush()
            current_start = _ts_to_seconds(int(m.group("m1")), int(m.group("s1")))
            current_end = _ts_to_seconds(int(m.group("m2")), int(m.group("s2")))
            current_label = (
                (m.group("dash_label") or "").strip().rstrip("(").strip()
                or (m.group("bare_label") or "").strip()
                or (m.group("paren_label") or "").strip()
            )
            continue

        if _SKIP_RE.match(line):
            collecting_osd = False
            continue

        os_m = _ONSCREEN_RE.match(line)
        if not os_m:
            os_m = _ONSCREEN_BRACKET_RE.match(line)
        if os_m:
            osd_val = _strip_quotes(os_m.group("text").strip())
            if osd_val:
                onscreen_buf.append(osd_val)
            collecting_osd = False
            continue

        if _ONSCREEN_BARE_RE.match(line):
            collecting_osd = True
            continue

        stripped = line.strip()
        if collecting_osd:
            if stripped:
                onscreen_buf.append(_strip_quotes(stripped))
                continue
            else:
                collecting_osd = False
                continue

        if stripped:
            cleaned = _VO_PREFIX_RE.sub("", stripped)
            narration_buf.append(_strip_quotes(cleaned))

    _flush()
    return sections


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_script(raw_text: str) -> list[ScriptSection]:
    """Parse *raw_text* and return an ordered list of :class:`ScriptSection`.

    Uses an LLM (``gpt-4o-mini``) as the primary parser so that any
    reasonable script format is supported.  Falls back to regex parsing
    when no API key is available or the LLM call fails.

    Raises ``ValueError`` if no timestamped sections can be found.
    """

    # 1. Try LLM first.
    sections = _parse_with_llm(raw_text)
    if sections:
        return sections

    # 2. Fallback to regex.
    sections = _parse_with_regex(raw_text)
    if sections:
        print(f"[script_parser] Regex fallback parsed {len(sections)} sections")
        return sections

    raise ValueError(
        "Could not find any timestamped sections in the script. "
        "Make sure each section has a timestamp range like 0:00–0:04."
    )
