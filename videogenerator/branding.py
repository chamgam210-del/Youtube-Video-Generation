from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class BrandingAssets:
    logo: Path
    intro: Path
    outro: Path


_SCHEMES: dict[str, dict[str, tuple[int, int, int]]] = {
    "orange": {"bg": (12, 12, 14), "accent": (249, 115, 22), "fg": (245, 245, 245)},
    "teal": {"bg": (10, 12, 14), "accent": (20, 184, 166), "fg": (245, 245, 245)},
    "purple": {"bg": (12, 10, 16), "accent": (168, 85, 247), "fg": (245, 245, 245)},
    "red": {"bg": (14, 10, 10), "accent": (239, 68, 68), "fg": (245, 245, 245)},
    "slate": {"bg": (11, 14, 18), "accent": (148, 163, 184), "fg": (245, 245, 245)},
    "lime": {"bg": (10, 14, 10), "accent": (132, 204, 22), "fg": (245, 245, 245)},
    "mono": {"bg": (12, 12, 12), "accent": (230, 230, 230), "fg": (255, 255, 255)},

    # Requested: orange + black font variants.
    # 1) Dark badge with orange BR letters + black outline.
    "orange_black": {"bg": (12, 12, 14), "accent": (249, 115, 22), "fg": (249, 115, 22), "stroke": (0, 0, 0)},
    # 2) Orange badge with black BR letters.
    "black_orange": {"bg": (249, 115, 22), "accent": (0, 0, 0), "fg": (0, 0, 0), "stroke": (255, 255, 255)},

    # More mix-and-match options emphasizing black letterforms.
    # 3) Dark badge: black BR letters outlined in orange.
    "black_orange_outline": {"bg": (12, 12, 14), "accent": (249, 115, 22), "fg": (0, 0, 0), "stroke": (249, 115, 22)},
    # 4) Dark badge: black BR letters outlined in white.
    "black_white_outline": {"bg": (12, 12, 14), "accent": (249, 115, 22), "fg": (0, 0, 0), "stroke": (255, 255, 255)},
    # 5) Orange badge: flat black BR letters (no outline).
    "black_orange_flat": {"bg": (249, 115, 22), "accent": (0, 0, 0), "fg": (0, 0, 0)},
}


def _try_load_font(*, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Prefer common Windows fonts, then DejaVu, then fallback.
    candidates = [
        "arialbd.ttf",
        "arial.ttf",
        "segoeuib.ttf",
        "segoeui.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _fit_font_for_box(
    *,
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
    start_size: int,
    min_size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(min_size, int(start_size))
    while size >= min_size:
        font = _try_load_font(size=size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        if tw <= max_width and th <= max_height:
            return font
        size -= max(1, int(size * 0.04))
    return _try_load_font(size=min_size)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int, max_lines: int) -> list[str]:
    words = (text or "").strip().split()
    if not words:
        return [""]

    lines: list[str] = []
    cur: list[str] = []

    def width_of(s: str) -> int:
        left, top, right, bottom = draw.textbbox((0, 0), s, font=font)
        return int(right - left)

    for w in words:
        trial = " ".join(cur + [w])
        if cur and width_of(trial) > max_width:
            lines.append(" ".join(cur))
            cur = [w]
            if len(lines) >= max_lines:
                break
        else:
            cur.append(w)

    if len(lines) < max_lines and cur:
        lines.append(" ".join(cur))

    # If we overflowed, ellipsize the last line.
    if len(lines) > max_lines:
        lines = lines[:max_lines]

    if len(lines) == max_lines and (" ".join(words) != " ".join(lines).strip()):
        last = lines[-1]
        while last and width_of(last + "…") > max_width:
            last = last[:-1].rstrip()
        lines[-1] = (last + "…") if last else "…"

    return lines


def _linear_gradient(size: tuple[int, int], *, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), color=top)
    px = img.load()
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        for x in range(w):
            px[x, y] = (r, g, b)
    return img


def create_br_logo(*, out_png: Path, size: int = 512, scheme: str = "orange") -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    s = _SCHEMES.get(str(scheme).strip().lower(), _SCHEMES["orange"])
    bg = s["bg"]
    accent = s["accent"]
    white = s["fg"]
    stroke = s.get("stroke")

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = int(size * 0.06)
    ring_w = max(2, int(size * 0.04))
    draw.ellipse((pad, pad, size - pad, size - pad), fill=bg + (255,), outline=accent + (255,), width=ring_w)

    # Subtle inner ring
    inner_pad = pad + int(size * 0.04)
    draw.ellipse((inner_pad, inner_pad, size - inner_pad, size - inner_pad), outline=(255, 255, 255, 35), width=max(1, int(size * 0.012)))

    text = "BR"
    # Classic logo sizing (keeps some breathing room inside the badge).
    font = _try_load_font(size=int(size * 0.38))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2
    y = (size - th) // 2 - int(size * 0.02)

    # Slight shadow for readability
    if stroke is not None:
        # Prefer stroke for high-contrast two-tone letters.
        try:
            draw.text(
                (x, y),
                text,
                font=font,
                fill=white + (255,),
                stroke_width=max(2, int(size * 0.03)),
                stroke_fill=tuple(stroke) + (255,),
            )
        except TypeError:
            # Pillow older fallback: shadow then fill.
            draw.text((x + 2, y + 2), text, font=font, fill=tuple(stroke) + (200,))
            draw.text((x, y), text, font=font, fill=white + (255,))
    else:
        # Slight shadow for readability
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 120))
        draw.text((x, y), text, font=font, fill=white + (255,))

    img.save(out_png)
    return out_png


def create_br_logo_variants(*, out_dir: Path, size: int = 512) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for scheme in _SCHEMES.keys():
        p = out_dir / f"brand_logo_br_{scheme}.png"
        if not p.exists():
            create_br_logo(out_png=p, size=size, scheme=scheme)
        out[scheme] = p
    return out


def regenerate_br_logo_variants(*, out_dir: Path, size: int = 512) -> dict[str, Path]:
    """Force-regenerate all logo variants (useful after design tweaks)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for scheme in _SCHEMES.keys():
        p = out_dir / f"brand_logo_br_{scheme}.png"
        create_br_logo(out_png=p, size=size, scheme=scheme)
        out[scheme] = p
    return out


def create_br_logo_filled(*, out_png: Path, size: int = 512, scheme: str = "orange") -> Path:
    """Alternate logo style where the letters fill the inner badge more tightly."""
    out_png.parent.mkdir(parents=True, exist_ok=True)

    s = _SCHEMES.get(str(scheme).strip().lower(), _SCHEMES["orange"])
    bg = s["bg"]
    accent = s["accent"]
    white = s["fg"]
    stroke = s.get("stroke")

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = int(size * 0.06)
    ring_w = max(2, int(size * 0.04))
    draw.ellipse((pad, pad, size - pad, size - pad), fill=bg + (255,), outline=accent + (255,), width=ring_w)

    inner_pad = pad + int(size * 0.04)
    draw.ellipse((inner_pad, inner_pad, size - inner_pad, size - inner_pad), outline=(255, 255, 255, 35), width=max(1, int(size * 0.012)))

    text = "BR"
    content_pad = inner_pad + int(size * 0.04)
    content_w = max(1, (size - content_pad) - content_pad)
    content_h = max(1, (size - content_pad) - content_pad)
    font = _fit_font_for_box(
        draw=draw,
        text=text,
        max_width=int(content_w * 0.98),
        max_height=int(content_h * 0.92),
        start_size=int(size * 0.60),
        min_size=int(size * 0.20),
    )

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2
    y = (size - th) // 2 - int(size * 0.02)

    if stroke is not None:
        try:
            draw.text(
                (x, y),
                text,
                font=font,
                fill=white + (255,),
                stroke_width=max(2, int(size * 0.03)),
                stroke_fill=tuple(stroke) + (255,),
            )
        except TypeError:
            draw.text((x + 2, y + 2), text, font=font, fill=tuple(stroke) + (200,))
            draw.text((x, y), text, font=font, fill=white + (255,))
    else:
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 120))
        draw.text((x, y), text, font=font, fill=white + (255,))

    img.save(out_png)
    return out_png


def _create_slate(
    *,
    out_png: Path,
    width: int,
    height: int,
    channel_name: str,
    title: str,
    logo_path: Path,
    mode: str,
) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    base = _linear_gradient((width, height), top=(10, 10, 12), bottom=(24, 20, 18)).convert("RGBA")
    draw = ImageDraw.Draw(base)

    accent = (249, 115, 22, 255)
    white = (245, 245, 245, 255)
    muted = (200, 200, 205, 255)

    # Accent stripe
    stripe_h = max(6, int(height * 0.012))
    draw.rectangle((0, 0, width, stripe_h), fill=accent)

    # Place logo
    logo = Image.open(logo_path).convert("RGBA")
    logo_size = int(min(width, height) * 0.22)
    logo = logo.resize((logo_size, logo_size), resample=Image.LANCZOS)

    logo_x = int(width * 0.07)
    logo_y = int(height * 0.16)
    base.alpha_composite(logo, dest=(logo_x, logo_y))

    # Text
    channel_font = _try_load_font(size=int(height * 0.06))
    title_font = _try_load_font(size=int(height * 0.08))
    small_font = _try_load_font(size=int(height * 0.045))

    text_x = logo_x + logo_size + int(width * 0.05)
    top_y = int(height * 0.18)

    # Channel name
    draw.text((text_x, top_y), channel_name, font=channel_font, fill=muted)

    # Title (wrapped)
    title_y = top_y + int(height * 0.10)
    max_title_w = int(width * 0.80) - text_x
    lines = _wrap_text(draw, title, title_font, max_width=max_title_w, max_lines=2)
    for i, line in enumerate(lines):
        draw.text((text_x, title_y + i * int(height * 0.095)), line, font=title_font, fill=white)

    if mode == "outro":
        outro_y = int(height * 0.70)
        draw.text((logo_x, outro_y), "Thanks for watching.", font=small_font, fill=white)
        draw.text((logo_x, outro_y + int(height * 0.06)), "Subscribe for more brutally honest reviews.", font=small_font, fill=muted)
        draw.text((logo_x, outro_y + int(height * 0.12)), "Comment what I should review next.", font=small_font, fill=muted)

    base.save(out_png)
    return out_png


def create_branding_assets(
    *,
    out_dir: Path,
    width: int,
    height: int,
    channel_name: str,
    title: str,
    logo_scheme: str = "orange",
) -> dict[str, Path]:
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Always create all logo variants so the user can pick later.
    variants = create_br_logo_variants(out_dir=assets_dir, size=512)

    chosen = str(logo_scheme).strip().lower()
    logo = variants.get(chosen, variants["orange"])

    # Ensure the selected logo reflects the current design (classic/original).
    try:
        create_br_logo(out_png=logo, size=512, scheme=chosen)
    except Exception:
        pass
    intro = assets_dir / "brand_intro.png"
    outro = assets_dir / "brand_outro.png"

    # Always regenerate slates since title/channel may change.
    _create_slate(
        out_png=intro,
        width=width,
        height=height,
        channel_name=channel_name,
        title=title,
        logo_path=logo,
        mode="intro",
    )
    _create_slate(
        out_png=outro,
        width=width,
        height=height,
        channel_name=channel_name,
        title=title,
        logo_path=logo,
        mode="outro",
    )

    return {"logo": logo, "intro": intro, "outro": outro}
