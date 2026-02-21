from __future__ import annotations

import json
import os
import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .models import Slide, TranscriptSegment


@dataclass(frozen=True)
class YouTubePackage:
    title: str
    description: str
    tags: list[str]
    thumbnail_slide_index: int
    verdict_label: str
    thumbnail_stamp_text: str | None = None
    # Optional: long-review thumbnail hook text + crop (debug/trace).
    thumbnail_text: str | None = None
    thumbnail_crop: dict[str, float] | None = None


def _openai_chat_completions(*, api_key: str, model: str, messages: list[dict[str, Any]], timeout_s: int = 60, max_retries: int = 5) -> str:
    import time as _time

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
    }
    for _attempt in range(max_retries + 1):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        if resp.status_code == 429 and _attempt < max_retries:
            _wait = min(int(resp.headers.get("Retry-After", 0)) or (30 * (_attempt + 1)), 120)
            print(f"[LLM] 429 rate-limited, retrying in {_wait}s (attempt {_attempt + 1}/{max_retries})")
            _time.sleep(_wait)
            continue
        resp.raise_for_status()
        break
    data = resp.json()
    return data["choices"][0]["message"]["content"]


_TITLE_STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "without",
    "movie",
    "film",
    "review",
}


def _sanitize_thumbnail_phrase(raw: str, *, title: str) -> str:
    s = " ".join(str(raw or "").split()).strip()
    if not s:
        return "WORTH IT?"

    # If the model returned multiple lines/quotes, take the first non-empty line.
    if "\n" in s:
        s = next((ln.strip() for ln in s.splitlines() if ln.strip()), "").strip()
    s = s.strip("\"'“”‘’` ")

    # Keep only letters/numbers/spaces, plus a single trailing ? or !.
    wants_q = "?" in s
    wants_bang = ("!" in s) and (not wants_q)
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s)
    s = " ".join(s.split()).strip()

    words = s.split()
    if len(words) > 4:
        words = words[:4]

    # Avoid repeating the title (drop significant title tokens if they appear).
    title_tokens = [
        t
        for t in re.sub(r"[^A-Za-z0-9\s]", " ", str(title or "")).lower().split()
        if t and (t not in _TITLE_STOP_WORDS) and (len(t) >= 4)
    ]
    if title_tokens and words:
        lowered = [w.lower() for w in words]
        filtered: list[str] = []
        for w in words:
            if w.lower() in title_tokens:
                continue
            filtered.append(w)
        if filtered:
            words = filtered

    if not words:
        out = "WORTH IT"
    else:
        out = " ".join(words)

    out = out.upper().strip()
    if wants_q:
        out = out.rstrip("?!") + "?"
    elif wants_bang:
        out = out.rstrip("?!") + "!"

    # Guardrails: avoid incomplete/auxiliary-only phrases that look broken on thumbnails.
    try:
        out_cmp = re.sub(r"[^A-Z0-9\s]", "", out).strip()
        toks = [t for t in out_cmp.split() if t]
        if toks == ["I", "WAS"]:
            return "I WAS WRONG"

        bad_two_word = {
            "I AM",
            "IM",
            "I WAS",
            "WE ARE",
            "WE WERE",
            "IT IS",
            "IT WAS",
            "THIS IS",
            "THAT WAS",
        }
        if out_cmp in bad_two_word:
            return "WORTH IT?"

        if len(toks) == 2 and all(len(t) <= 3 for t in toks) and (not (out.endswith("?") or out.endswith("!"))):
            return "WORTH IT?"
        if len(toks) == 1 and len(toks[0]) <= 3 and (not (out.endswith("?") or out.endswith("!"))):
            return "WORTH IT?"
    except Exception:
        pass
    return out


def pick_review_thumbnail_text_with_llm(
    segments: list[TranscriptSegment],
    *,
    title: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Pick one CTR-optimized thumbnail phrase for a long movie review.

    Uses the user's production prompt. Returns a sanitized 1–4 word phrase.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "WORTH IT?"

    transcript = " ".join((str(s.text or "").strip() for s in segments if str(s.text or "").strip())).strip()
    if not transcript:
        return "WORTH IT?"

    # Keep within a reasonable size so requests don't blow up.
    if len(transcript) > 18000:
        transcript = transcript[:18000]

    system = (
        "You are a YouTube growth expert specializing in movie review channels.\n\n"
        "Your task is to select ONE thumbnail text (2–4 words max) that maximizes click-through rate.\n\n"
        "This text must:\n\n"
        "Be emotionally charged\n\n"
        "Create curiosity or controversy\n\n"
        "Be readable on a phone\n\n"
        "NOT summarize the review\n\n"
        "NOT repeat the title\n\n"
        "NOT use punctuation beyond “?” or “!”\n\n"
        "The thumbnail text should feel like a reaction, not an explanation.\n\n"
        "Do NOT explain your reasoning.\n\n"
        "Output ONLY the final thumbnail text, nothing else."
    )

    user = (
        f"Movie title: {str(title or '').strip()}\n\n"
        "Transcript:\n"
        f"{transcript}\n\n"
        "Guidelines:\n"
        "- Choose ONE phrase (2–4 words max)\n"
        "- Prioritize curiosity over accuracy\n"
        "- Prefer emotional or opinionated language\n"
        "- If the review is mixed or hesitant, lean controversial\n"
        "- If the review is positive but unexpected, lean surprise\n"
        "- If the review is negative, lean disappointment or disbelief\n\n"
        "Examples of good outputs:\n"
        "WORTH IT?\n"
        "SURPRISINGLY GOOD\n"
        "I WAS WRONG\n"
        "THIS WORKS\n"
        "WHAT HAPPENED?\n"
        "NOT WHAT I EXPECTED\n\n"
        "Return ONLY the thumbnail text."
    )

    try:
        out = _openai_chat_completions(
            api_key=api_key,
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            timeout_s=60,
        )
    except Exception:
        return "WORTH IT?"

    return _sanitize_thumbnail_phrase(out, title=title)


def generate_youtube_package(
    segments: list[TranscriptSegment] | None,
    *,
    slides: list[Slide],
    topic: str | None,
    channel_name: str,
    title: str,
    video_type: str = "review",
    model: str = "gpt-4o-mini",
) -> YouTubePackage:
    """Generate a YouTube title/description/tags plus thumbnail choice.

    Best-effort: uses LLM if OPENAI_API_KEY is present, otherwise falls back.
    """

    if not slides:
        raise ValueError("slides must be non-empty")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and segments:
        try:
            return _generate_with_llm(
                api_key=api_key,
                segments=segments,
                slides=slides,
                topic=topic,
                channel_name=channel_name,
                title=title,
                video_type=video_type,
                model=model,
            )
        except Exception:
            pass

    description = _fallback_description(segments=segments, topic=topic, channel_name=channel_name, title=title)
    tags = _fallback_tags(topic=topic)
    vt = (video_type or "review").strip().lower()
    thumb_idx = _fallback_thumbnail_index(slides)
    verdict = _fallback_verdict_label(segments=segments) if vt == "review" else ""
    stamp = _fallback_stamp_text(topic=topic, title=title) if vt in {"explainer", "shorts"} else None
    return YouTubePackage(
        title=title,
        description=description,
        tags=tags,
        thumbnail_slide_index=thumb_idx,
        verdict_label=verdict,
        thumbnail_stamp_text=stamp,
    )


def _generate_with_llm(
    *,
    api_key: str,
    segments: list[TranscriptSegment],
    slides: list[Slide],
    topic: str | None,
    channel_name: str,
    title: str,
    video_type: str,
    model: str,
) -> YouTubePackage:
    transcript = [
        {"start": round(float(s.start), 2), "end": round(float(s.end), 2), "text": str(s.text)}
        for s in segments
    ]

    slide_choices = []
    for i, s in enumerate(slides[:50]):
        slide_choices.append({"i": i, "query": (s.query or ""), "filename": Path(s.image_path).name})

    system = (
        "You are a YouTube producer. Create metadata based on the transcript. "
        "If the provided topic hint conflicts with the transcript, ignore the topic hint. "
        "Do NOT spoil the final verdict, rating, or conclusion. "
        "Avoid wording like 'I loved it'/'I hated it' or 'the verdict is'. "
        "Instead, tease themes/topics and invite viewers to watch for the final take. "
        "Return ONLY valid JSON (no markdown). "
        "Schema: {"
        "\"description\": <string>, "
        "\"tags\": <array of strings>, "
        "\"thumbnail_slide_index\": <int>, "
        "\"verdict_label\": <string>, "
        "\"thumbnail_stamp_text\": <string|null>"
        "}. "
        "Description should be 2-4 short paragraphs, include a brief hook, and a subtle CTA. "
        "Tags: 10-18 items, no hashtags, keep them short. "
        "thumbnail_slide_index must be one of the provided slide choice i values; "
        "prefer a slide that is likely to show a character/actor (stills, cast) based on its query. "
        "verdict_label MUST be exactly one of: 'Masterpiece!', 'Decent', 'Mehhh!', 'Garbage!' OR empty string ''. "
        "thumbnail_stamp_text MUST be null OR exactly one of: 'TOP THEORIES', 'EXPLAINED', 'BREAKDOWN', 'DEEP DIVE'. "
        "If this is a review video: set verdict_label to one of the review labels and set thumbnail_stamp_text=null. "
        "If this is an explainer/list/theories video: set verdict_label='' and set thumbnail_stamp_text to the best stamp label. "
        "Decide based on the transcript and the provided video_type hint."
    )

    user = {
        "channel_name": channel_name,
        "topic": topic or "",
        "title": title,
        "video_type": (video_type or "review"),
        "transcript": transcript,
        "slide_choices": slide_choices,
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

    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM returned non-object JSON")

    desc = str(parsed.get("description") or "").strip()
    tags = parsed.get("tags") or []
    idx = int(parsed.get("thumbnail_slide_index"))
    verdict = str(parsed.get("verdict_label") or "").strip()
    stamp = parsed.get("thumbnail_stamp_text")
    stamp_text = None if stamp is None else str(stamp).strip()

    if not desc:
        raise RuntimeError("LLM returned empty description")
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise RuntimeError("LLM returned invalid tags")
    tags = [t.strip() for t in tags if t.strip()]
    if idx < 0 or idx >= len(slides):
        idx = _fallback_thumbnail_index(slides)

    vt = (video_type or "review").strip().lower()
    if vt == "review":
        if verdict not in {"Masterpiece!", "Decent", "Mehhh!", "Garbage!"}:
            verdict = _fallback_verdict_label(segments=segments)
        stamp_text = None
    else:
        verdict = ""
        allowed = {"TOP THEORIES", "EXPLAINED", "BREAKDOWN", "DEEP DIVE"}
        if stamp_text not in allowed:
            stamp_text = _fallback_stamp_text(topic=topic, title=title)

    return YouTubePackage(
        title=title,
        description=desc,
        tags=tags,
        thumbnail_slide_index=idx,
        verdict_label=verdict,
        thumbnail_stamp_text=stamp_text,
    )


def write_youtube_metadata_text(out_dir: str | Path, pkg: YouTubePackage) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = out_dir / "youtube_metadata.txt"
    tags_csv = ", ".join(pkg.tags)

    text = (
        f"Title:\n{pkg.title}\n\n"
        f"Description:\n{pkg.description.strip()}\n\n"
        f"Tags:\n{tags_csv}\n\n"
        f"Thumbnail slide index:\n{pkg.thumbnail_slide_index}\n\n"
        f"Thumbnail verdict label:\n{pkg.verdict_label}\n"
        f"Thumbnail stamp text:\n{pkg.thumbnail_stamp_text or ''}\n"
        f"Thumbnail text:\n{pkg.thumbnail_text or ''}\n"
        f"Thumbnail crop:\n{json.dumps(pkg.thumbnail_crop or {}, ensure_ascii=False)}\n"
    )
    p.write_text(text, encoding="utf-8")
    return p


def create_thumbnail(
    *,
    out_path: str | Path,
    background_image: str | Path,
    text: str = "Brutally Honest Review",
    verdict_text: str | None = None,
    stamp_text: str | None = None,
    width: int = 1920,
    height: int = 1080,
    match_video_frame: bool = False,
    theme: str = "default",
    show_title: bool = True,
    crop: dict[str, float] | None = None,
    title_scale: float = 1.0,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(background_image) as im:
        im = im.convert("RGB")

        pre_cropped_to_aspect = False

        # Optional pre-crop (normalized coordinates in [0,1]) before aspect fitting.
        if crop and all(k in crop for k in ("x", "y", "w", "h")):
            try:
                cx = float(crop.get("x") or 0.0)
                cy = float(crop.get("y") or 0.0)
                cw = float(crop.get("w") or 0.0)
                ch = float(crop.get("h") or 0.0)
                if cw > 0.01 and ch > 0.01:
                    src_w, src_h = im.size
                    x0 = int(max(0.0, min(1.0, cx)) * src_w)
                    y0 = int(max(0.0, min(1.0, cy)) * src_h)
                    x1 = int(max(0.0, min(1.0, cx + cw)) * src_w)
                    y1 = int(max(0.0, min(1.0, cy + ch)) * src_h)

                    # Expand the crop box to match the target aspect ratio by growing (not shrinking)
                    # around the crop center when possible. This helps prevent faces being clipped
                    # by a second cover-crop step.
                    target_ratio = width / float(height)
                    box_w = max(2, x1 - x0)
                    box_h = max(2, y1 - y0)
                    cx_px = x0 + box_w / 2.0
                    cy_px = y0 + box_h / 2.0
                    cur_ratio = box_w / float(box_h)

                    if abs(cur_ratio - target_ratio) > 1e-3:
                        if cur_ratio > target_ratio:
                            # Too wide: expand height.
                            new_h = int(round(box_w / float(target_ratio)))
                            new_w = box_w
                        else:
                            # Too tall: expand width.
                            new_w = int(round(box_h * float(target_ratio)))
                            new_h = box_h

                        new_w = max(2, min(int(src_w), int(new_w)))
                        new_h = max(2, min(int(src_h), int(new_h)))

                        nx0 = int(round(cx_px - new_w / 2.0))
                        ny0 = int(round(cy_px - new_h / 2.0))
                        nx1 = nx0 + new_w
                        ny1 = ny0 + new_h

                        # Clamp box into bounds while preserving size.
                        if nx0 < 0:
                            nx1 -= nx0
                            nx0 = 0
                        if ny0 < 0:
                            ny1 -= ny0
                            ny0 = 0
                        if nx1 > src_w:
                            shift = nx1 - src_w
                            nx0 -= shift
                            nx1 = src_w
                        if ny1 > src_h:
                            shift = ny1 - src_h
                            ny0 -= shift
                            ny1 = src_h

                        nx0 = max(0, int(nx0))
                        ny0 = max(0, int(ny0))
                        nx1 = min(int(src_w), int(nx1))
                        ny1 = min(int(src_h), int(ny1))
                    else:
                        nx0, ny0, nx1, ny1 = x0, y0, x1, y1

                    if nx1 > nx0 + 2 and ny1 > ny0 + 2:
                        im = im.crop((nx0, ny0, nx1, ny1))
                        pre_cropped_to_aspect = True
            except Exception:
                pass

        theme_norm = (theme or "default").strip().lower()

        if match_video_frame:
            # Match render.py: scale down to fit + pad (no cropping), so the thumbnail background
            # matches the first video frame as closely as possible.
            src_w, src_h = im.size
            scale = min(width / float(src_w), height / float(src_h))
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))
            im_resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (width, height), (0, 0, 0))
            x = (width - new_w) // 2
            y = (height - new_h) // 2
            canvas.paste(im_resized, (x, y))
            im = canvas
        else:
            # Cover-crop to target aspect (more cinematic thumbnail look).
            src_w, src_h = im.size
            target_ratio = width / float(height)
            src_ratio = src_w / float(src_h)

            if src_ratio > target_ratio:
                # too wide
                new_w = int(src_h * target_ratio)
                left = (src_w - new_w) // 2
                im = im.crop((left, 0, left + new_w, src_h))
            else:
                # too tall
                new_h = int(src_w / target_ratio)
                # For long-review thumbnails, bias upward to leave headroom so the face isn't under the title.
                excess = max(0, src_h - new_h)
                if theme_norm == "review_long":
                    top = int(round(excess * 0.32))
                else:
                    top = excess // 2
                im = im.crop((0, top, src_w, top + new_h))

            im = im.resize((width, height), Image.Resampling.LANCZOS)

            # Slight contrast + darken for text legibility.
            # Skip for review_long — it applies its own darken pass.
            if theme_norm != "review_long":
                im = ImageEnhance.Contrast(im).enhance(1.05)
                overlay = Image.new("RGB", (width, height), (0, 0, 0))
                im = Image.blend(im, overlay, alpha=0.18)

        # If we already pre-cropped to (approximately) the target aspect, avoid an additional
        # cover-crop effect by just resizing to the output size.
        if pre_cropped_to_aspect and not match_video_frame:
            try:
                im = im.resize((width, height), Image.Resampling.LANCZOS)
            except Exception:
                pass

        # Long review thumbnail: title + verdict stamp (reference-style), but with the
        # title slightly smaller to avoid dominating the frame.
        if theme_norm == "review_long":
            # Darken for contrast — lighter than before since title is hidden by default.
            im = ImageEnhance.Contrast(im).enhance(1.10)
            overlay = Image.new("RGB", (width, height), (0, 0, 0))
            im = Image.blend(im, overlay, alpha=0.15)

            # Subtle vignette: draws attention to centre without over-darkening edges.
            try:
                vignette = Image.new("L", (width, height), 0)
                vd = ImageDraw.Draw(vignette)
                pad_x = int(width * 0.10)
                pad_y = int(height * 0.12)
                vd.ellipse((pad_x, pad_y, width - pad_x, height - pad_y), fill=255)
                vignette = vignette.filter(ImageFilter.GaussianBlur(radius=int(min(width, height) * 0.06)))
                vignette = Image.eval(vignette, lambda a: 255 - a)
                shade = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                shade.putalpha(vignette.point(lambda a: int(a * 0.22)))
                base = im.convert("RGBA")
                base.alpha_composite(shade)
                im = base.convert("RGB")
            except Exception:
                pass

            draw = ImageDraw.Draw(im)

            # Title: when no verdict stamp, centre vertically and render BIG;
            # otherwise keep the original top-positioned smaller title.
            has_verdict = bool(verdict_text and str(verdict_text).strip() and str(verdict_text).strip().lower() != "decent")
            if show_title:
                title_text = " ".join((text or "").split()).strip()
                if title_text:
                    title_stroke_w = max(4, width // 180)
                    title_fill = (0, 0, 0)  # black
                    title_stroke = (255, 140, 0)  # orange outline

                    if has_verdict:
                        # Title at top — larger than before.
                        _max_lines = 3
                        _max_h_ratio = 0.28
                        _max_size = min(180, int(width * 0.12))
                        title_stroke_w = max(6, width // 130)
                    else:
                        # No stamp → big centred title.
                        _max_lines = 2
                        _max_h_ratio = 0.40
                        _max_size = min(260, int(width * 0.18))
                        title_stroke_w = max(8, width // 100)

                    fitted_text, title_font = _fit_title_text(
                        draw,
                        title_text,
                        width=width,
                        height=height,
                        stroke_width=title_stroke_w,
                        max_lines=_max_lines,
                        max_text_h_ratio=_max_h_ratio,
                        max_size=_max_size,
                    )
                    spacing = max(6, int(getattr(title_font, "size", 64) * 0.12))
                    title_bbox = draw.multiline_textbbox(
                        (0, 0),
                        fitted_text,
                        font=title_font,
                        align="center",
                        stroke_width=title_stroke_w,
                        spacing=spacing,
                    )
                    title_w = title_bbox[2] - title_bbox[0]
                    title_h = title_bbox[3] - title_bbox[1]
                    title_x = int((width - title_w) / 2)
                    if has_verdict:
                        top_margin = max(14, title_stroke_w * 2)
                        title_y = int(top_margin)
                    else:
                        # Vertically centre.
                        title_y = int((height - title_h) / 2)
                    draw.multiline_text(
                        (title_x, title_y),
                        fitted_text,
                        font=title_font,
                        fill=title_fill,
                        align="center",
                        stroke_width=title_stroke_w,
                        stroke_fill=title_stroke,
                        spacing=spacing,
                    )

            # Verdict stamp (bottom).
            chosen_stamp = (str(verdict_text).strip() if verdict_text else "")
            if chosen_stamp and chosen_stamp.lower() != "decent":
                vt = chosen_stamp
                # Green verdict text (no backing plate), slightly smaller.
                verdict_font = _pick_font(vt, width, size_hint=int(width * 0.105), max_size=210)
                verdict_stroke_w = max(9, width // 90)
                verdict_fill = (0, 200, 83, 255)
                verdict_stroke = (0, 0, 0, 255)
                verdict_cx = int(width * 0.5)
                verdict_cy = int(height * 0.86)

                stamp_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                stamp_draw = ImageDraw.Draw(stamp_layer)
                stamp_spacing = max(6, int(getattr(verdict_font, "size", 64) * 0.10))
                stamp_bbox = stamp_draw.multiline_textbbox(
                    (0, 0),
                    vt,
                    font=verdict_font,
                    align="center",
                    stroke_width=verdict_stroke_w,
                    spacing=stamp_spacing,
                )
                stamp_w = stamp_bbox[2] - stamp_bbox[0]
                stamp_h = stamp_bbox[3] - stamp_bbox[1]
                stamp_x = int(verdict_cx - stamp_w / 2)
                stamp_y = int(verdict_cy - stamp_h / 2)

                stamp_draw.multiline_text(
                    (stamp_x, stamp_y),
                    vt,
                    font=verdict_font,
                    fill=verdict_fill,
                    align="center",
                    stroke_width=verdict_stroke_w,
                    stroke_fill=verdict_stroke,
                    spacing=stamp_spacing,
                )

                # Tilt the stamp layer slightly for a casual / punchy look.
                stamp_layer = stamp_layer.rotate(
                    -6, resample=Image.BICUBIC, expand=False,
                    center=(verdict_cx, verdict_cy),
                )

                base = im.convert("RGBA")
                base.alpha_composite(stamp_layer)
                im = base.convert("RGB")

            im.save(out_path, format="PNG")
            return out_path

        draw = ImageDraw.Draw(im)

        # Visual rule: if a review is merely "Decent", show no stamp at all.
        if verdict_text is not None and str(verdict_text).strip().lower() == "decent":
            verdict_text = None
        if theme_norm in {"shorts", "highlight", "highlight_pills"}:
            # Reference-style: highlighted title + stamp boxes.
            _draw_highlight_title_and_stamp(
                im,
                title=(text if show_title else ""),
                stamp=(str(verdict_text).strip() if verdict_text else (str(stamp_text).strip() if stamp_text else "")),
            )
        else:
            # Main title: auto-wrap and fit so long titles don't clip.
            title_stroke_w = max(4, width // 160)
            title_fill = (0, 0, 0)  # black
            title_stroke = (255, 140, 0)  # orange outline

            title_cx = int(width * 0.5)
            top_margin = max(18, title_stroke_w * 2)
            title_cy = int(top_margin)

            _title_max = int(min(150, int(width * 0.095)) * max(0.5, float(title_scale)))
            fitted_text, title_font = _fit_title_text(
                draw,
                text,
                width=width,
                height=height,
                stroke_width=title_stroke_w,
                max_lines=3,
                max_size=_title_max,
            )

            spacing = max(6, int(getattr(title_font, "size", 64) * 0.12))
            # Pillow compatibility: some versions don't support `anchor` for multiline text.
            title_bbox = draw.multiline_textbbox(
                (0, 0),
                fitted_text,
                font=title_font,
                align="center",
                stroke_width=title_stroke_w,
                spacing=spacing,
            )
            title_w = title_bbox[2] - title_bbox[0]
            title_x = int(title_cx - title_w / 2)
            title_y = int(title_cy)
            draw.multiline_text(
                (title_x, title_y),
                fitted_text,
                font=title_font,
                fill=title_fill,
                align="center",
                stroke_width=title_stroke_w,
                stroke_fill=title_stroke,
                spacing=spacing,
            )

        # Default theme adds the mid-frame stamp.
        if theme_norm not in {"shorts", "highlight", "highlight_pills"}:
            # Stamp: either verdict (reviews) or a generic explainer stamp.
            chosen_stamp = (str(verdict_text).strip() if verdict_text else "")
            if not chosen_stamp and stamp_text:
                chosen_stamp = str(stamp_text).strip()

            if chosen_stamp:
                vt = chosen_stamp
                if vt:
                    # Stamp.
                    # Green for verdict, warm yellow for explainer.
                    is_verdict = vt in {"Masterpiece!", "Mehhh!", "Garbage!"}

                    # Reviews: stamp should sit near the bottom, no tilt, smaller but higher contrast.
                    if is_verdict:
                        verdict_font = _pick_font(vt, width, size_hint=int(width * 0.115), max_size=220)
                        verdict_stroke_w = max(10, width // 85)
                    else:
                        verdict_font = _pick_font(vt, width, size_hint=int(width * 0.165), max_size=280)
                        verdict_stroke_w = max(8, width // 75)

                    verdict_fill = (0, 200, 83, 255) if is_verdict else (255, 215, 0, 255)
                    verdict_stroke = (0, 0, 0, 255)  # black outline

                    verdict_cx = int(width * 0.5)
                    verdict_cy = int(height * (0.86 if is_verdict else 0.54))
                    angle_deg = 0 if is_verdict else 12  # reviews: no tilt; others keep stamped feel

                    # Draw on a separate transparent layer, rotate, then composite.
                    stamp_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                    stamp_draw = ImageDraw.Draw(stamp_layer)
                    stamp_spacing = max(6, int(getattr(verdict_font, "size", 64) * 0.10))
                    stamp_bbox = stamp_draw.multiline_textbbox(
                        (0, 0),
                        vt,
                        font=verdict_font,
                        align="center",
                        stroke_width=verdict_stroke_w,
                        spacing=stamp_spacing,
                    )
                    stamp_w = stamp_bbox[2] - stamp_bbox[0]
                    stamp_h = stamp_bbox[3] - stamp_bbox[1]
                    stamp_x = int(verdict_cx - stamp_w / 2)
                    stamp_y = int(verdict_cy - stamp_h / 2)

                    # Backing plate for better contrast (especially on busy backgrounds).
                    stamp_draw.multiline_text(
                        (stamp_x, stamp_y),
                        vt,
                        font=verdict_font,
                        fill=verdict_fill,
                        align="center",
                        stroke_width=verdict_stroke_w,
                        stroke_fill=verdict_stroke,
                        spacing=stamp_spacing,
                    )

                    # Crop to the drawn content + padding so rotation doesn't clip.
                    bbox = stamp_layer.getbbox()
                    if bbox:
                        pad = max(18, verdict_stroke_w * 2)
                        l = max(0, bbox[0] - pad)
                        t = max(0, bbox[1] - pad)
                        r = min(width, bbox[2] + pad)
                        b = min(height, bbox[3] + pad)
                        stamp_cropped = stamp_layer.crop((l, t, r, b))
                    else:
                        stamp_cropped = stamp_layer

                    stamp_rot = stamp_cropped.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=True)

                    # Slight transparency for a stamped feel.
                    if stamp_rot.mode != "RGBA":
                        stamp_rot = stamp_rot.convert("RGBA")
                    alpha = stamp_rot.split()[-1]
                    alpha_scale = 1.0 if is_verdict else 0.92
                    alpha = alpha.point(lambda a: int(a * alpha_scale))
                    stamp_rot.putalpha(alpha)

                    # Paste rotated stamp centered at (verdict_cx, verdict_cy).
                    paste_x = int(verdict_cx - stamp_rot.size[0] / 2)
                    paste_y = int(verdict_cy - stamp_rot.size[1] / 2)

                    base = im.convert("RGBA")
                    base.alpha_composite(stamp_rot, dest=(paste_x, paste_y))
                    im = base.convert("RGB")

        im.save(out_path, format="PNG")

    return out_path


def _encode_image_data_url(path: str | Path, *, max_side: int = 512, quality: int = 78) -> str:
    p = Path(path)
    with Image.open(p) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, float(max_side) / float(max(w, h)))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(quality), optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _encode_pil_image_data_url(im: Image.Image, *, max_side: int = 512, quality: int = 78) -> str:
    im = im.convert("RGB")
    w, h = im.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality), optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _vision_verify_centered_person(
    *,
    api_key: str,
    model: str,
    image_path: Path,
    crop: dict[str, float] | None,
    timeout_s: int = 60,
) -> bool:
    """Return True if (after applying crop) the image has a clearly visible centered person.

    Close-ups are OK; we just need a visible person/face as the dominant subject,
    and that subject should be near the center of the frame.
    """

    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            if crop and all(k in crop for k in ("x", "y", "w", "h")):
                try:
                    cx = float(crop.get("x") or 0.0)
                    cy = float(crop.get("y") or 0.0)
                    cw = float(crop.get("w") or 0.0)
                    ch = float(crop.get("h") or 0.0)
                    if cw > 0.01 and ch > 0.01:
                        src_w, src_h = im.size
                        x0 = int(max(0.0, min(1.0, cx)) * src_w)
                        y0 = int(max(0.0, min(1.0, cy)) * src_h)
                        x1 = int(max(0.0, min(1.0, cx + cw)) * src_w)
                        y1 = int(max(0.0, min(1.0, cy + ch)) * src_h)
                        if x1 > x0 + 2 and y1 > y0 + 2:
                            im = im.crop((x0, y0, x1, y1))
                except Exception:
                    pass

            url = _encode_pil_image_data_url(im, max_side=512, quality=78)

        system = (
            "You are a strict thumbnail QA checker. Return ONLY valid JSON (no markdown).\n"
            "Schema: {\"ok\": <bool>}\n"
            "ok must be true ONLY if ALL are satisfied:\n"
            "- There is a clearly visible PERSON/CHARACTER as the dominant subject (not a landscape/object).\n"
            "- The person is near the center of the frame (roughly centered).\n"
            "- The face is a CLOSE-UP (head/face fills a significant portion of the frame).\n"
            "- The face shows an INTENSE or dramatic expression (anger, shock, determination, emotion) — NOT neutral/blank.\n"
            "- The face is visible (eyes visible) OR it is an obvious close-up of the person.\n"
            "- The face/head is NOT cut off by the frame edges (no missing forehead/chin/cheeks due to cropping).\n"
            "- Composition leaves headroom: the face/head should not be too high in frame (reserve space for a title at the top).\n"
            "- Not a poster/collage/text-heavy graphic.\n"
        )

        content = _openai_chat_completions(
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Check this candidate."},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                },
            ],
            timeout_s=int(timeout_s),
        )

        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return bool(parsed.get("ok"))
        return False
    except Exception:
        return False


def _vision_detect_main_face_bbox(
    *,
    api_key: str,
    model: str,
    image_path: Path,
    timeout_s: int = 60,
) -> dict[str, float] | None:
    """Detect the dominant visible face bbox in normalized coordinates.

    Returns {x,y,w,h} in [0,1] or None if not found.
    """

    try:
        url = _encode_image_data_url(image_path, max_side=512, quality=78)
        system = (
            "You are a precise vision detector. Return ONLY valid JSON (no markdown).\n"
            "Schema: {\"found\": <bool>, \"bbox\": {\"x\":<float>,\"y\":<float>,\"w\":<float>,\"h\":<float>}}\n"
            "Rules:\n"
            "- If a face is visible, choose the MOST PROMINENT face (largest/most central).\n"
            "- bbox must tightly enclose the full face/head (include forehead+chin; don’t crop it).\n"
            "- Coordinates are normalized [0,1] with x,y as TOP-LEFT.\n"
            "- If no clear face is visible, set found=false." 
        )

        content = _openai_chat_completions(
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Detect the main face bbox."},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                },
            ],
            timeout_s=int(timeout_s),
        )

        parsed = json.loads(content)
        if not isinstance(parsed, dict) or not bool(parsed.get("found")):
            return None
        bbox = parsed.get("bbox")
        if not isinstance(bbox, dict):
            return None
        x = float(bbox.get("x"))
        y = float(bbox.get("y"))
        w = float(bbox.get("w"))
        h = float(bbox.get("h"))
        if w <= 0.01 or h <= 0.01:
            return None
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.01, min(1.0 - x, w))
        h = max(0.01, min(1.0 - y, h))
        return {"x": x, "y": y, "w": w, "h": h}
    except Exception:
        return None


def _crop_center_face(
    *,
    face_bbox: dict[str, float],
    target_ratio: float,
    face_fill: float = 0.62,
    headroom_frac: float = 0.18,
) -> dict[str, float]:
    """Compute a crop window that centers the face and leaves top headroom for title.

    - face_fill ~ fraction of crop height covered by face bbox height.
    - headroom_frac ~ reserved top portion (for title), pushing face lower.
    """

    fx = float(face_bbox.get("x"))
    fy = float(face_bbox.get("y"))
    fw = float(face_bbox.get("w"))
    fh = float(face_bbox.get("h"))
    fcx = fx + fw / 2.0
    fcy = fy + fh / 2.0

    # Choose crop height such that the face occupies ~face_fill of crop height.
    crop_h = min(1.0, max(0.20, fh / max(0.20, min(0.90, face_fill))))
    crop_w = min(1.0, max(0.20, crop_h * float(target_ratio)))

    # If width is constrained, recompute height from width.
    if crop_w >= 0.999 and target_ratio > 0:
        crop_h = min(1.0, crop_w / float(target_ratio))

    # Place face near center, but slightly lower to leave room for title at top.
    desired_x = 0.50
    # Map headroom into desired face Y position inside crop.
    desired_y = min(0.70, max(0.45, 0.50 + float(headroom_frac) * 0.55))

    x = fcx - desired_x * crop_w
    y = fcy - desired_y * crop_h

    # Clamp.
    x = max(0.0, min(1.0 - crop_w, x))
    y = max(0.0, min(1.0 - crop_h, y))

    return {"x": float(x), "y": float(y), "w": float(crop_w), "h": float(crop_h)}


def pick_long_review_thumbnail_with_vision(
    *,
    slides: list[Slide],
    topic: str | None,
    title: str,
    model: str = "gpt-4o-mini",
    max_candidates: int = 8,
    forced_text: str | None = None,
) -> tuple[int, str, dict[str, float] | None]:
    """Pick (slide_index, thumbnail_text, crop) for long review thumbnails.

    Uses OpenAI vision if OPENAI_API_KEY is set; otherwise falls back.
    crop is normalized {x,y,w,h} in [0,1] relative to the selected image.
    """

    if not slides:
        return 0, "WORTH IT?", None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_thumbnail_index(slides), "WORTH IT?", None

    forced = " ".join(str(forced_text or "").split()).strip()
    if forced:
        words = forced.split()
        if len(words) > 4:
            forced = " ".join(words[:4])

    # Prefer likely character/scene stills; avoid poster-y queries when possible.
    scored: list[tuple[int, float]] = []
    for i, s in enumerate(slides[:50]):
        q = (s.query or "").lower()
        score = 0.0
        if "poster" in q:
            score -= 2.0
        if any(k in q for k in ("still", "scene", "screencap", "frame", "cast")):
            score += 1.0
        if any(k in q for k in ("close", "portrait", "face", "headshot")):
            score += 1.2
        if any(k in q for k in ("intense", "angry", "emotional", "dramatic", "expression", "stare", "scream", "crying")):
            score += 1.0
        scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Build a small, diverse candidate set (top scored + a few spaced picks) to increase
    # odds of finding a clear face close-up.
    max_c = max(4, min(int(max_candidates), len(scored)))
    top = [i for i, _ in scored[: max(2, max_c // 2)]]
    spaced: list[int] = []
    if len(scored) > 6:
        step = max(1, len(scored) // max(3, max_c // 2))
        for j in range(0, len(scored), step):
            spaced.append(scored[j][0])
            if len(spaced) >= max(2, max_c - len(top)):
                break

    picks: list[int] = []
    for i in (top + spaced):
        if i not in picks:
            picks.append(i)
        if len(picks) >= max_c:
            break
    if not picks:
        picks = [0]

    system = (
        "You are selecting a YouTube MOVIE REVIEW thumbnail background. "
        "Follow these rules strictly:\n"
        "- The image MUST feature a clearly visible PERSON/CHARACTER as the dominant subject.\n"
        "- That person MUST be near the center of the frame (centered composition).\n"
        "- STRONGLY PREFER close-up shots showing the face/head filling most of the frame.\n"
        "- The character should have an INTENSE, dramatic, or emotional facial expression "
        "(e.g. anger, shock, determination, fear, pain). Avoid neutral/blank expressions.\n"
        "- Face/eyes MUST be visible; avoid tiny full-body or wide shots.\n"
        "- Do NOT crop so tight that any part of the face/head is cut off. Leave a little breathing room.\n"
        "- Use ONLY one focal point (one person). Avoid crowds.\n"
        "- Avoid posters, collages, text-heavy images, logos.\n"
        "- Provide ONE strong text phrase (2-4 words max). Examples: 'WORTH IT?', 'SURPRISING', 'I WAS WRONG', 'BRUTAL'.\n"
        "Return ONLY valid JSON (no markdown).\n"
        "Schema: {\"index\": <int>, \"text\": <string>, \"crop\": {\"x\":<float>,\"y\":<float>,\"w\":<float>,\"h\":<float>}}\n"
        "crop must be normalized [0,1] (top-left x,y + w,h) and should tightly frame the face/head with some breathing room."
    )

    def _pad_crop(c: dict[str, float] | None, *, pad_frac: float = 0.12) -> dict[str, float] | None:
        if not c:
            return None
        try:
            x = float(c.get("x"))
            y = float(c.get("y"))
            w = float(c.get("w"))
            h = float(c.get("h"))
        except Exception:
            return c
        if w <= 0.01 or h <= 0.01:
            return c

        # Expand around center to reduce risk of cutting off forehead/chin.
        cx = x + w / 2.0
        cy = y + h / 2.0
        w2 = min(1.0, w * (1.0 + float(pad_frac)))
        h2 = min(1.0, h * (1.0 + float(pad_frac)))
        x2 = cx - w2 / 2.0
        y2 = cy - h2 / 2.0

        # Clamp into [0,1].
        x2 = max(0.0, min(1.0 - w2, x2))
        y2 = max(0.0, min(1.0 - h2, y2))
        return {"x": float(x2), "y": float(y2), "w": float(w2), "h": float(h2)}

    def _add_headroom(c: dict[str, float] | None, *, headroom_frac_of_h: float = 0.10) -> dict[str, float] | None:
        """Shift crop up a bit so the subject lands lower (more top headroom)."""
        if not c:
            return None
        try:
            x = float(c.get("x"))
            y = float(c.get("y"))
            w = float(c.get("w"))
            h = float(c.get("h"))
        except Exception:
            return c
        if h <= 0.01:
            return c
        y2 = max(0.0, y - float(headroom_frac_of_h) * h)
        y2 = min(1.0 - h, y2)
        return {"x": x, "y": float(y2), "w": w, "h": h}

    def _pick_from_candidates(candidate_idxs: list[int]) -> tuple[int, str, dict[str, float] | None]:
        user_text = {
            "title": (title or "").strip(),
            "topic": (topic or "").strip(),
            "forced_text": forced,
            "candidates": [
                {"i": idx, "query": (slides[idx].query or ""), "filename": Path(slides[idx].image_path).name}
                for idx in candidate_idxs
            ],
        }

        content_parts: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(user_text, ensure_ascii=False)}]
        for idx in candidate_idxs:
            try:
                p = Path(slides[idx].image_path)
                try:
                    raw = _try_find_raw_for_card(p)
                    if raw is not None:
                        p = raw
                except Exception:
                    pass
                url = _encode_image_data_url(p)
                content_parts.append({"type": "image_url", "image_url": {"url": url}})
            except Exception:
                continue

        content = _openai_chat_completions(
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content_parts},
            ],
            timeout_s=90,
        )

        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise RuntimeError("Vision thumbnail picker returned non-object JSON")

        idx = int(parsed.get("index"))
        if idx not in candidate_idxs:
            idx = candidate_idxs[0]

        text = " ".join(str(parsed.get("text") or "").split()).strip() or "WORTH IT?"
        if forced:
            text = forced
        words = text.split()
        if len(words) > 4:
            text = " ".join(words[:4])

        crop = parsed.get("crop")
        crop_out: dict[str, float] | None = None
        if isinstance(crop, dict):
            try:
                cx = float(crop.get("x"))
                cy = float(crop.get("y"))
                cw = float(crop.get("w"))
                ch = float(crop.get("h"))
                if cw > 0.01 and ch > 0.01:
                    crop_out = {
                        "x": max(0.0, min(1.0, cx)),
                        "y": max(0.0, min(1.0, cy)),
                        "w": max(0.01, min(1.0, cw)),
                        "h": max(0.01, min(1.0, ch)),
                    }
            except Exception:
                crop_out = None

        crop_out = _pad_crop(crop_out, pad_frac=0.14)
        crop_out = _add_headroom(crop_out, headroom_frac_of_h=0.12)

        return idx, text, crop_out

    remaining = list(picks)
    best_idx, best_text, best_crop = remaining[0], "WORTH IT?", None
    for _attempt in range(3):
        if not remaining:
            break
        idx, text, crop_out = _pick_from_candidates(remaining)
        best_idx, best_text, best_crop = idx, text, crop_out

        # Verify: must be a centered visible person (close-up ok) AND face should be centered.
        try:
            p = Path(slides[idx].image_path)
            raw = None
            try:
                raw = _try_find_raw_for_card(p)
            except Exception:
                raw = None
            if raw is not None:
                p = raw

            # Use face bbox to override crop for better centering.
            face_bbox = _vision_detect_main_face_bbox(api_key=api_key, model=model, image_path=p)
            if face_bbox is not None:
                target_ratio = 16.0 / 9.0
                crop_out = _pad_crop(_crop_center_face(face_bbox=face_bbox, target_ratio=target_ratio), pad_frac=0.10)

            if _vision_verify_centered_person(api_key=api_key, model=model, image_path=p, crop=crop_out):
                return best_idx, best_text, crop_out
        except Exception:
            pass

        remaining = [i for i in remaining if i != idx]

    return best_idx, best_text, best_crop


def overlay_shorts_title_and_stamp(
    slides: list[Slide],
    *,
    out_dir: str | Path,
    title: str,
    stamp_text: str | None,
    footer_text: str | None = None,
    width: int,
    height: int,
    show_title: bool = True,
    prefer_raw_first_scene: bool = True,
) -> list[Slide]:
    """For Shorts: bake a persistent overlay (stamp, optionally title) onto each slide image."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_label = (stamp_text or "").strip()
    # Visual rule: if a review verdict is "Decent", show no stamp.
    if raw_label.lower() == "decent":
        label = ""
    else:
        label = raw_label or "TOP THEORIES"
    base_title = " ".join((title or "").split()).strip() if show_title else ""

    out: list[Slide] = []
    for i, s in enumerate(slides):
        try:
            src = Path(s.image_path)
            # Special case: for the first scene, prefer the raw downloaded image (s00_*.png)
            # instead of the card_00.png (which contains LLM scene text).
            if i == 0 and bool(prefer_raw_first_scene):
                raw0 = _try_find_raw_for_card(src)
                if raw0 is not None:
                    src = raw0
            dst = out_dir / f"slide_{i:03d}.png"

            cur_title = base_title
            _overlay_highlight_pills(
                in_path=src,
                out_path=dst,
                width=width,
                height=height,
                title=cur_title,
                stamp=label,
                footer=str(footer_text or "").strip(),
            )
            out.append(
                Slide(
                    start=float(s.start),
                    end=float(s.end),
                    image_path=str(dst),
                    query=s.query,
                    source_page=s.source_page,
                    image_url=s.image_url,
                    license_name=s.license_name,
                    license_url=s.license_url,
                    attribution=s.attribution,
                )
            )
        except Exception:
            out.append(s)
    return out


def _try_find_raw_for_card(card_path: Path) -> Path | None:
    """Given assets/card_XX.png, try to find assets/sXX_*.png."""

    try:
        name = card_path.name
        if not name.startswith("card_") or not name.endswith(".png"):
            return None
        idx = int(name.replace("card_", "").replace(".png", ""))
    except Exception:
        return None

    parent = card_path.parent
    # The raw downloads are named like s00_*.png.
    prefix = f"s{idx:02d}_"
    try:
        cands = sorted(parent.glob(prefix + "*.png"))
    except Exception:
        cands = []
    if not cands:
        return None
    # Prefer the last one (most recent) in case there are multiple.
    return cands[-1]


def _overlay_highlight_pills(
    *,
    in_path: Path,
    out_path: Path,
    width: int,
    height: int,
    title: str,
    stamp: str,
    footer: str = "",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(in_path) as im:
        im = im.convert("RGB")
        if im.size != (int(width), int(height)):
            # Shorts requirement: fill the entire frame (no letterboxing).
            # Cover-crop to target aspect, then resize.
            src_w, src_h = im.size
            target_ratio = float(width) / float(height)
            src_ratio = float(src_w) / float(src_h)

            if src_ratio > target_ratio:
                # too wide: crop left/right
                new_w = int(round(src_h * target_ratio))
                left = max(0, (src_w - new_w) // 2)
                im = im.crop((left, 0, left + new_w, src_h))
            else:
                # too tall: crop top/bottom
                new_h = int(round(src_w / target_ratio))
                top = max(0, (src_h - new_h) // 2)
                im = im.crop((0, top, src_w, top + new_h))

            im = im.resize((int(width), int(height)), Image.Resampling.LANCZOS)

        base = im.convert("RGBA")
        _draw_highlight_pills_on_image(base, title=title, stamp=stamp, footer=footer)
        base.convert("RGB").save(out_path, format="PNG")


def _fit_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    max_width: int,
    max_height: int,
    max_lines: int,
    size_max: int,
    size_min: int,
) -> tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    raw = " ".join((text or "").split()).strip()
    if not raw:
        return "", _load_font_with_size(size_min)

    step = 4
    best_font: ImageFont.FreeTypeFont | ImageFont.ImageFont = _load_font_with_size(size_min)
    best_wrapped = raw

    for size in range(int(size_max), int(size_min) - 1, -step):
        font = _load_font_with_size(size)
        wrapped = _wrap_text_to_width(
            draw,
            raw,
            font=font,
            max_width=int(max_width),
            stroke_width=0,
            max_lines=int(max_lines),
        )
        if not wrapped:
            continue
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center")
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= int(max_width) and h <= int(max_height):
            return wrapped, font
        best_font = font
        best_wrapped = wrapped

    return best_wrapped, best_font


def _draw_highlight_title_and_stamp(
    im: Image.Image,
    *,
    title: str,
    stamp: str,
) -> None:
    """Draw the Shorts-style highlight title+stamp directly onto an image."""

    base = im.convert("RGBA")
    _draw_highlight_pills_on_image(base, title=title, stamp=stamp, footer="")
    im.paste(base.convert("RGB"))


def _draw_highlight_pills_on_image(base: Image.Image, *, title: str, stamp: str, footer: str) -> None:
    draw = ImageDraw.Draw(base, "RGBA")

    w, h = base.size
    top_y = int(h * 0.05)
    bottom_y = int(h * 0.78)

    title_bg = (255, 214, 0, 235)  # warm yellow
    stamp_bg = (66, 214, 78, 235)  # vivid green
    text_fill = (255, 255, 255, 255)
    shadow_fill = (0, 0, 0, 95)

    def _draw_highlighted_line(
        *,
        line: str,
        font: ImageFont.ImageFont,
        cx: int,
        y: int,
        fill: tuple[int, int, int, int],
    ) -> int:
        if not line.strip():
            return int(y)

        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=0)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        x = int(cx - tw / 2)
        pad_x = max(10, int(getattr(font, "size", 64) * 0.45))
        pad_y = max(8, int(getattr(font, "size", 64) * 0.22))

        rx0 = max(0, x - pad_x)
        ry0 = max(0, int(y) - pad_y)
        rx1 = min(w, x + int(tw) + pad_x)
        ry1 = min(h, int(y) + int(th) + pad_y)

        r = max(14, int((ry1 - ry0) * 0.32))
        # shadow highlight
        draw.rounded_rectangle((rx0 + 4, ry0 + 5, rx1 + 4, ry1 + 5), radius=r, fill=shadow_fill)
        # highlight behind letters
        draw.rounded_rectangle((rx0, ry0, rx1, ry1), radius=r, fill=fill)

        # text shadow (use a top-aligned anchor to match bbox math)
        draw.text((cx + 2, int(y) + 3), line, font=font, fill=(0, 0, 0, 140), anchor="mt")
        draw.text((cx, int(y)), line, font=font, fill=text_fill, anchor="mt")
        return int(y) + int(getattr(font, "size", 64) * 1.05)

    # Title: wrap and draw line-by-line with per-line highlight.
    max_title_w = int(w * 0.92)
    max_title_h = int(h * 0.28)
    title_wrapped, title_font = _fit_text_in_box(
        draw,
        title,
        max_width=max_title_w,
        max_height=max_title_h,
        max_lines=3,
        size_max=min(170, int(w * 0.12)),
        size_min=44,
    )
    title_lines = [ln for ln in (title_wrapped or "").splitlines() if ln.strip()]
    cur_y = int(top_y)
    gap = max(6, int(getattr(title_font, "size", 64) * 0.22))
    for idx, ln in enumerate(title_lines):
        cur_y = _draw_highlighted_line(line=ln, font=title_font, cx=w // 2, y=cur_y, fill=title_bg)
        if idx < len(title_lines) - 1:
            cur_y += int(gap)

    # Stamp: draw near bottom with per-line highlight.
    stamp = " ".join((stamp or "").split()).strip()
    if stamp:
        stamp_wrapped, stamp_font = _fit_text_in_box(
            draw,
            stamp,
            max_width=int(w * 0.92),
            max_height=int(h * 0.22),
            max_lines=2,
            size_max=min(200, int(w * 0.16)),
            size_min=54,
        )
        stamp_lines = [ln for ln in (stamp_wrapped or "").splitlines() if ln.strip()]
        # Start slightly above bottom_y so it doesn't clip.
        sy = int(bottom_y)
        sgap = max(6, int(getattr(stamp_font, "size", 64) * 0.14))
        for idx, ln in enumerate(stamp_lines):
            sy = _draw_highlighted_line(line=ln, font=stamp_font, cx=w // 2, y=sy, fill=stamp_bg)
            if idx < len(stamp_lines) - 1:
                sy += int(sgap)

    footer = " ".join((footer or "").split()).strip()
    if footer:
        footer_scale = 0.75
        footer_wrapped, footer_font = _fit_text_in_box(
            draw,
            footer,
            max_width=int(w * 0.92),
            max_height=int(h * 0.10),
            max_lines=2,
            size_max=max(18, int(min(78, int(w * 0.075)) * footer_scale)),
            size_min=max(14, int(34 * footer_scale)),
        )
        footer_lines = [ln for ln in (footer_wrapped or "").splitlines() if ln.strip()]
        # Keep it comfortably above the bottom safe area.
        fy = int(h * 0.865)
        fgap = max(4, int(getattr(footer_font, "size", 44) * 0.16))
        for idx, ln in enumerate(footer_lines):
            fy = _draw_highlighted_line(line=ln, font=footer_font, cx=w // 2, y=fy, fill=title_bg)
            if idx < len(footer_lines) - 1:
                fy += int(fgap)


def _pick_font(
    text: str,
    width: int,
    *,
    size_hint: int | None = None,
    max_size: int = 140,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Try common bold fonts on Windows; fall back to PIL default.
    font_candidates = [
        r"C:\\Windows\\Fonts\\impact.ttf",
        r"C:\\Windows\\Fonts\\arialbd.ttf",
        r"C:\\Windows\\Fonts\\seguisb.ttf",
        r"C:\\Windows\\Fonts\\segoeuib.ttf",
    ]

    # Rough sizing target.
    base = int(width * 0.085)
    if size_hint is not None:
        base = int(size_hint)
    size = max(48, min(int(max_size), base))

    for fp in font_candidates:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            continue

    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _load_font_with_size(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        r"C:\\Windows\\Fonts\\impact.ttf",
        r"C:\\Windows\\Fonts\\arialbd.ttf",
        r"C:\\Windows\\Fonts\\seguisb.ttf",
        r"C:\\Windows\\Fonts\\segoeuib.ttf",
    ]

    for fp in font_candidates:
        try:
            return ImageFont.truetype(fp, size=int(size))
        except Exception:
            continue

    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    stroke_width: int,
    max_lines: int,
) -> str:
    raw = " ".join((text or "").split())
    if not raw:
        return ""

    words = raw.split(" ")
    lines: list[str] = []
    cur = ""

    def _fits(s: str) -> bool:
        if not s:
            return True
        bbox = draw.textbbox((0, 0), s, font=font, stroke_width=stroke_width)
        return (bbox[2] - bbox[0]) <= int(max_width)

    for w in words:
        test = (cur + " " + w).strip() if cur else w
        if _fits(test):
            cur = test
            continue

        if cur:
            lines.append(cur)
            cur = ""
            if len(lines) >= max_lines:
                break

        # If a single word doesn't fit, hard-break it.
        if not _fits(w):
            chunk = ""
            for ch in w:
                t2 = chunk + ch
                if _fits(t2):
                    chunk = t2
                else:
                    if chunk:
                        lines.append(chunk)
                        if len(lines) >= max_lines:
                            chunk = ""
                            break
                    chunk = ch
            if chunk and len(lines) < max_lines:
                cur = chunk
        else:
            cur = w

    if cur and len(lines) < max_lines:
        lines.append(cur)

    return "\n".join(lines)


def _fit_title_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    width: int,
    height: int,
    stroke_width: int,
    max_lines: int,
    max_text_h_ratio: float = 0.28,
    max_size: int | None = None,
) -> tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    # Safe area near the top so we don't collide with the mid-frame stamp.
    max_text_w = int(width * 0.92)
    max_text_h = int(height * float(max_text_h_ratio))

    # Start big and step down until it fits.
    max_size_eff = int(max_size) if max_size is not None else min(150, int(width * 0.095))
    min_size = 44
    step = 4

    best_text = " ".join((text or "").split())
    best_font: ImageFont.FreeTypeFont | ImageFont.ImageFont = _pick_font(best_text, width)

    for size in range(max_size_eff, min_size - 1, -step):
        font = _load_font_with_size(size)
        wrapped = _wrap_text_to_width(
            draw,
            best_text,
            font=font,
            max_width=max_text_w,
            stroke_width=stroke_width,
            max_lines=max_lines,
        )
        if not wrapped:
            continue

        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center", stroke_width=stroke_width)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        if w <= max_text_w and h <= max_text_h:
            return wrapped, font

        best_font = font

    # Final fallback: truncate to last line if needed.
    wrapped = _wrap_text_to_width(
        draw,
        best_text,
        font=best_font,
        max_width=max_text_w,
        stroke_width=stroke_width,
        max_lines=max_lines,
    )
    return wrapped or best_text, best_font


def _fallback_description(
    *,
    segments: list[TranscriptSegment] | None,
    topic: str | None,
    channel_name: str,
    title: str,
) -> str:
    # Prefer the title; topic hints can be stale/wrong.
    t = (title or "").strip()
    if not t:
        t = (topic or "this episode").strip() or "this episode"

    # Keep it intentionally non-committal (no verdict).
    return (
        f"Today on {channel_name}, we’re breaking down {t} — what works, what doesn’t, and why it’s got people talking.\n\n"
        "We’ll touch on the premise, the tone, the performances, and the craft behind the scenes — without giving away the big take too early. "
        "Watch to the end for the full perspective.\n\n"
        "What was your biggest takeaway? Drop it in the comments."
    )


def _fallback_tags(*, topic: str | None) -> list[str]:
    base = ["review", "tv review", "analysis", "explained", "spoiler free", "apple tv plus"]
    if topic:
        base.insert(0, topic.replace("(TV series)", "").strip())
    # De-dup
    out: list[str] = []
    seen: set[str] = set()
    for t in base:
        t = t.strip()
        if not t:
            continue
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
    return out


def _fallback_thumbnail_index(slides: list[Slide]) -> int:
    # Avoid the very first frame (often an establishing shot); pick an early-middle one.
    if not slides:
        return 0
    return min(len(slides) - 1, max(0, len(slides) // 3))


def _fallback_verdict_label(*, segments: list[TranscriptSegment] | None) -> str:
    # Conservative default when LLM isn't available: mixed/neutral.
    text = ""
    if segments:
        text = " ".join(s.text for s in segments[-20:]).lower()
    # Heuristic: very basic polarity hints.
    pos = any(w in text for w in ("masterpiece", "amazing", "incredible", "fantastic", "excellent", "love it", "loved it"))
    neg = any(w in text for w in ("garbage", "terrible", "awful", "hate it", "hated it", "worst", "boring"))
    if pos and not neg:
        return "Masterpiece!"
    if neg and not pos:
        return "Garbage!"
    # Neutral/mixed: treat as "Decent" (explicitly requested).
    return "Decent"


def _fallback_stamp_text(*, topic: str | None, title: str) -> str:
    t = (title or "").lower()
    if "theor" in t:
        return "TOP THEORIES"
    if "top" in t or "rank" in t:
        return "BREAKDOWN"
    if topic and "theor" in (topic or "").lower():
        return "TOP THEORIES"
    return "EXPLAINED"
