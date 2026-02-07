from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont


@dataclass(frozen=True)
class SlideCardSpec:
    headline: str
    subhead: str | None = None


def render_slide_card(
    *,
    background_image: str | Path,
    out_path: str | Path,
    spec: SlideCardSpec,
    width: int,
    height: int,
) -> Path:
    """Render a text-on-image slide card.

    The returned image is designed to be used directly as a slideshow slide.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(background_image) as im:
        im = im.convert("RGB")

        # Cover-crop to target aspect.
        src_w, src_h = im.size
        target_ratio = width / float(height)
        src_ratio = src_w / float(src_h)

        if src_ratio > target_ratio:
            new_w = int(src_h * target_ratio)
            left = (src_w - new_w) // 2
            im = im.crop((left, 0, left + new_w, src_h))
        else:
            new_h = int(src_w / target_ratio)
            top = (src_h - new_h) // 2
            im = im.crop((0, top, src_w, top + new_h))

        im = im.resize((width, height), Image.Resampling.LANCZOS)

        # Darken a bit for legibility.
        im = ImageEnhance.Contrast(im).enhance(1.05)
        overlay = Image.new("RGB", (width, height), (0, 0, 0))
        im = Image.blend(im, overlay, alpha=0.28)

        draw = ImageDraw.Draw(im)

        headline = (spec.headline or "").strip()
        subhead = (spec.subhead or "").strip() if spec.subhead else ""

        # Fonts.
        headline_font = _pick_font(headline, width, max_size=int(width * 0.085), min_size=54)
        subhead_font = _pick_font(subhead, width, max_size=int(width * 0.050), min_size=36)

        # Layout.
        margin_x = int(width * 0.07)
        y = int(height * 0.18)

        # Headline (wrapped).
        head_lines = _wrap_to_width(draw, headline, headline_font, max_width=width - 2 * margin_x)
        y = _draw_centered_lines(
            draw,
            lines=head_lines,
            font=headline_font,
            cx=width // 2,
            y=y,
            fill=(255, 255, 255),
            stroke_width=max(3, width // 320),
            stroke_fill=(0, 0, 0),
            line_gap=int(headline_font.size * 0.18),
        )

        # Subhead.
        if subhead:
            y += int(height * 0.03)
            sub_lines = _wrap_to_width(draw, subhead, subhead_font, max_width=width - 2 * margin_x)
            _draw_centered_lines(
                draw,
                lines=sub_lines,
                font=subhead_font,
                cx=width // 2,
                y=y,
                fill=(235, 235, 235),
                stroke_width=max(2, width // 420),
                stroke_fill=(0, 0, 0),
                line_gap=int(subhead_font.size * 0.22),
            )

        im.save(out_path, format="PNG")

    return out_path


def _pick_font(text: str, width: int, *, max_size: int, min_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        r"C:\\Windows\\Fonts\\impact.ttf",
        r"C:\\Windows\\Fonts\\arialbd.ttf",
        r"C:\\Windows\\Fonts\\seguisb.ttf",
        r"C:\\Windows\\Fonts\\segoeuib.ttf",
    ]

    size = max(min_size, min(int(max_size), int(width * 0.080)))

    for fp in font_candidates:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            continue

    return ImageFont.load_default()


def _wrap_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, *, max_width: int) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []

    words = t.split()
    lines: list[str] = []
    cur: list[str] = []

    def fits(s: str) -> bool:
        bbox = draw.textbbox((0, 0), s, font=font, stroke_width=0)
        return (bbox[2] - bbox[0]) <= max_width

    for w in words:
        test = " ".join(cur + [w])
        if not cur or fits(test):
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]

    if cur:
        lines.append(" ".join(cur))

    return lines


def _draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    *,
    lines: list[str],
    font: ImageFont.ImageFont,
    cx: int,
    y: int,
    fill: tuple[int, int, int],
    stroke_width: int,
    stroke_fill: tuple[int, int, int],
    line_gap: int,
) -> int:
    cur_y = int(y)
    for line in lines:
        if not line.strip():
            continue
        draw.text(
            (cx, cur_y),
            line,
            font=font,
            fill=fill,
            anchor="ma",
            stroke_width=int(stroke_width),
            stroke_fill=stroke_fill,
        )
        cur_y += int(font.size) + int(line_gap)
    return cur_y
