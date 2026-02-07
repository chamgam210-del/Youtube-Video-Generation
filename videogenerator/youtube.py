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
    thumb_idx = _fallback_thumbnail_index(slides)
    vt = (video_type or "review").strip().lower()
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
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(background_image) as im:
        im = im.convert("RGB")

        # Cover-crop to target aspect.
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

        title_font = _pick_font(text, width)

        # Main title: centered but slightly above middle.
        title_stroke_w = max(4, width // 160)
        title_fill = (0, 0, 0)  # black
        title_stroke = (255, 140, 0)  # orange outline

        # Pin the title to the top while keeping a small safe margin so stroke doesn't clip.
        title_cx = int(width * 0.5)
        top_margin = max(18, title_stroke_w * 2)
        title_cy = int(top_margin)
        draw.multiline_text(
            (title_cx, title_cy),
            text,
            font=title_font,
            fill=title_fill,
            align="center",
            anchor="mt",
            stroke_width=title_stroke_w,
            stroke_fill=title_stroke,
        )

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
                stamp_draw.multiline_text(
                    (verdict_cx, verdict_cy),
                    vt,
                    font=verdict_font,
                    fill=verdict_fill,
                    align="center",
                    anchor="mm",
                    stroke_width=verdict_stroke_w,
                    stroke_fill=verdict_stroke,
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
