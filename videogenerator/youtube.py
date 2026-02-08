from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from .models import Slide, TranscriptSegment


@dataclass(frozen=True)
class YouTubePackage:
    title: str
    description: str
    tags: list[str]
    thumbnail_slide_index: int
    verdict_label: str
    thumbnail_stamp_text: str | None = None


def _openai_chat_completions(*, api_key: str, model: str, messages: list[dict[str, Any]], timeout_s: int = 60) -> str:
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
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


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
        "verdict_label MUST be exactly one of: 'Masterpiece!', 'Mehhh!', 'Garbage!' OR empty string ''. "
        "thumbnail_stamp_text MUST be null OR exactly one of: 'TOP THEORIES', 'EXPLAINED', 'BREAKDOWN', 'DEEP DIVE'. "
        "If this is a review video: set verdict_label to one of the three labels and set thumbnail_stamp_text=null. "
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
        if verdict not in {"Masterpiece!", "Mehhh!", "Garbage!"}:
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
    width: int = 1280,
    height: int = 720,
    match_video_frame: bool = False,
    theme: str = "default",
    show_title: bool = True,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(background_image) as im:
        im = im.convert("RGB")

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
                top = (src_h - new_h) // 2
                im = im.crop((0, top, src_w, top + new_h))

            im = im.resize((width, height), Image.Resampling.LANCZOS)

            # Slight contrast + darken for text legibility.
            im = ImageEnhance.Contrast(im).enhance(1.05)
            overlay = Image.new("RGB", (width, height), (0, 0, 0))
            im = Image.blend(im, overlay, alpha=0.18)

        draw = ImageDraw.Draw(im)

        theme_norm = (theme or "default").strip().lower()
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

            fitted_text, title_font = _fit_title_text(
                draw,
                text,
                width=width,
                height=height,
                stroke_width=title_stroke_w,
                max_lines=3,
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
                    verdict_font = _pick_font(vt, width, size_hint=int(width * 0.165), max_size=280)
                    verdict_stroke_w = max(8, width // 75)
                    # Green for verdict, warm yellow for explainer.
                    is_verdict = vt in {"Masterpiece!", "Mehhh!", "Garbage!"}
                    verdict_fill = (0, 200, 83, 255) if is_verdict else (255, 215, 0, 255)
                    verdict_stroke = (0, 0, 0, 255)  # black outline

                    verdict_cx = int(width * 0.5)
                    verdict_cy = int(height * 0.54)
                    angle_deg = 12  # slight stamp tilt (opposite direction)

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
                    alpha = alpha.point(lambda a: int(a * 0.92))
                    stamp_rot.putalpha(alpha)

                    # Paste rotated stamp centered at (verdict_cx, verdict_cy).
                    paste_x = int(verdict_cx - stamp_rot.size[0] / 2)
                    paste_y = int(verdict_cy - stamp_rot.size[1] / 2)

                    base = im.convert("RGBA")
                    base.alpha_composite(stamp_rot, dest=(paste_x, paste_y))
                    im = base.convert("RGB")

        im.save(out_path, format="PNG")

    return out_path


def overlay_shorts_title_and_stamp(
    slides: list[Slide],
    *,
    out_dir: str | Path,
    title: str,
    stamp_text: str | None,
    width: int,
    height: int,
    show_title: bool = True,
) -> list[Slide]:
    """For Shorts: bake a persistent overlay (stamp, optionally title) onto each slide image."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label = (stamp_text or "").strip() or "TOP THEORIES"
    base_title = " ".join((title or "").split()).strip() if show_title else ""

    out: list[Slide] = []
    for i, s in enumerate(slides):
        try:
            src = Path(s.image_path)
            # Special case: for the first scene, prefer the raw downloaded image (s00_*.png)
            # instead of the card_00.png (which contains LLM scene text).
            if i == 0:
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
        _draw_highlight_pills_on_image(base, title=title, stamp=stamp)
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
    _draw_highlight_pills_on_image(base, title=title, stamp=stamp)
    im.paste(base.convert("RGB"))


def _draw_highlight_pills_on_image(base: Image.Image, *, title: str, stamp: str) -> None:
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

        # text shadow
        draw.text((cx + 2, int(y) + 3), line, font=font, fill=(0, 0, 0, 140), anchor="ma")
        draw.text((cx, int(y)), line, font=font, fill=text_fill, anchor="ma")
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
) -> tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    # Safe area near the top so we don't collide with the mid-frame stamp.
    max_text_w = int(width * 0.92)
    max_text_h = int(height * 0.28)

    # Start big and step down until it fits.
    max_size = min(160, int(width * 0.10))
    min_size = 44
    step = 4

    best_text = " ".join((text or "").split())
    best_font: ImageFont.FreeTypeFont | ImageFont.ImageFont = _pick_font(best_text, width)

    for size in range(max_size, min_size - 1, -step):
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
    return "Mehhh!"


def _fallback_stamp_text(*, topic: str | None, title: str) -> str:
    t = (title or "").lower()
    if "theor" in t:
        return "TOP THEORIES"
    if "top" in t or "rank" in t:
        return "BREAKDOWN"
    if topic and "theor" in (topic or "").lower():
        return "TOP THEORIES"
    return "EXPLAINED"
