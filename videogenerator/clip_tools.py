"""Search, download, trim, and prepare video clips from the web (YouTube, etc.)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio_ffmpeg

if TYPE_CHECKING:
    from .models import VideoClipSuggestion

def _ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


# ── SerpAPI video search ────────────────────────────────────────────────────


@dataclass
class VideoSearchResult:
    title: str
    url: str
    source: str  # "youtube", "vimeo", etc.
    duration_seconds: float | None = None
    thumbnail: str | None = None


def _parse_duration(text: str | None) -> float | None:
    """Parse duration strings like '2:34' or '1:02:30' to seconds."""
    if not text:
        return None
    parts = str(text).strip().split(":")
    try:
        parts = [int(p) for p in parts]
        if len(parts) == 2:
            return float(parts[0] * 60 + parts[1])
        if len(parts) == 3:
            return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
        if len(parts) == 1:
            return float(parts[0])
    except (ValueError, TypeError):
        pass
    return None


def search_video_clips(
    query: str,
    *,
    max_results: int = 5,
    preferred_max_duration: float = 300.0,
) -> list[VideoSearchResult]:
    """Search for video clips using SerpAPI YouTube search.

    Falls back to Google Video search if YouTube-specific search isn't available.
    """

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return []

    import requests

    results: list[VideoSearchResult] = []

    # Strategy 1: SerpAPI YouTube search.
    try:
        params = {
            "engine": "youtube",
            "search_query": query,
            "api_key": api_key,
        }
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("video_results", [])[:max_results * 2]:
            url = item.get("link") or ""
            title = item.get("title") or ""
            dur = _parse_duration(item.get("length", {}).get("text") if isinstance(item.get("length"), dict) else item.get("length"))
            thumb = None
            thumbnails = item.get("thumbnail", {})
            if isinstance(thumbnails, dict):
                thumb = thumbnails.get("static") or thumbnails.get("rich")
            elif isinstance(thumbnails, str):
                thumb = thumbnails

            if not url or "youtube.com" not in url:
                continue
            if dur and dur > preferred_max_duration:
                continue

            results.append(
                VideoSearchResult(
                    title=title,
                    url=url,
                    source="youtube",
                    duration_seconds=dur,
                    thumbnail=thumb,
                )
            )
            if len(results) >= max_results:
                break
    except Exception:
        pass

    # Strategy 2: If YouTube search returned nothing, try Google Video search.
    if not results:
        try:
            params = {
                "engine": "google_videos",
                "q": query,
                "api_key": api_key,
            }
            resp = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("video_results", [])[:max_results * 2]:
                url = item.get("link") or ""
                title = item.get("title") or ""
                dur_text = None
                rich = item.get("rich_snippet", {})
                if isinstance(rich, dict):
                    dur_text = rich.get("duration")
                dur = _parse_duration(dur_text)
                thumb = item.get("thumbnail", {})
                if isinstance(thumb, dict):
                    thumb = thumb.get("src")

                if not url:
                    continue
                if dur and dur > preferred_max_duration:
                    continue

                source = "youtube" if "youtube.com" in url or "youtu.be" in url else "web"
                results.append(
                    VideoSearchResult(
                        title=title,
                        url=url,
                        source=source,
                        duration_seconds=dur,
                        thumbnail=thumb if isinstance(thumb, str) else None,
                    )
                )
                if len(results) >= max_results:
                    break
        except Exception:
            pass

    return results


def _pick_best_clip_with_llm(
    candidates: list[VideoSearchResult],
    *,
    search_query: str,
    reason: str,
    model: str = "gpt-4o-mini",
) -> int:
    """Use the LLM to pick the best clip from search results. Returns index."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not candidates:
        return 0

    import requests

    compact = []
    for i, c in enumerate(candidates[:10]):
        compact.append({
            "i": i,
            "title": c.title[:120],
            "url": c.url,
            "source": c.source,
            "duration": c.duration_seconds,
        })

    system = (
        "You are selecting the best video clip for a movie review B-roll insertion.\n"
        "Pick the clip that best matches the search query and reason.\n"
        "Prefer: official trailers, movie scene clips, behind-the-scenes footage.\n"
        "Avoid: fan edits, reaction videos, unrelated content, full movies.\n"
        "Return ONLY valid JSON: {\"index\": <int>}"
    )
    user = {
        "search_query": search_query,
        "reason": reason,
        "candidates": compact,
    }

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ], "temperature": 0.2, "max_tokens": 100},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(content)
        idx = int(parsed.get("index", 0))
        if 0 <= idx < len(candidates):
            return idx
    except Exception:
        pass
    return 0


# ── Download via yt-dlp ─────────────────────────────────────────────────────


def _find_ytdlp() -> str | None:
    """Find yt-dlp executable. Returns None if not installed."""
    import shutil as sh

    for name in ("yt-dlp", "yt-dlp.exe"):
        p = sh.which(name)
        if p:
            return p
    return None


def download_clip(
    url: str,
    dest_dir: str | Path,
    *,
    prefix: str = "clip",
    max_duration: float = 60.0,
) -> Path | None:
    """Download a video clip using yt-dlp. Returns the downloaded file path, or None on failure."""

    ytdlp = _find_ytdlp()
    if not ytdlp:
        raise RuntimeError(
            "yt-dlp is not installed. Install it with: pip install yt-dlp  (or: uv pip install yt-dlp)"
        )

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    out_template = str(dest_dir / f"{prefix}_%(id)s.%(ext)s")

    # Point yt-dlp at the imageio-ffmpeg bundled binary so it can merge streams.
    # imageio-ffmpeg names the binary e.g. "ffmpeg-win-x86_64-v7.1.exe" — yt-dlp
    # expects "ffmpeg" or "ffmpeg.exe", so we create a shim copy/symlink.
    ffmpeg_path = Path(_ffmpeg_exe())
    ffmpeg_dir = str(ffmpeg_path.parent)
    expected_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    shim = ffmpeg_path.parent / expected_name
    if not shim.exists():
        try:
            shim.symlink_to(ffmpeg_path)
        except OSError:
            # Symlinks may require developer mode on Windows; fall back to copy.
            import shutil as _shutil
            _shutil.copy2(ffmpeg_path, shim)

    cmd = [
        ytdlp,
        "--no-playlist",
        "--ffmpeg-location", ffmpeg_dir,
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        "--max-filesize", "100M",
        "--socket-timeout", "30",
        "--retries", "2",
        "--no-overwrites",
        "--no-post-overwrites",
        # Skip download if longer than max_duration (yt-dlp supports this for some extractors).
        "--match-filter", f"duration<={int(max_duration)}",
        url,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return None

    if proc.returncode != 0:
        # Check if match-filter rejected it (expected for long videos).
        stderr = proc.stderr or ""
        if "does not pass filter" in stderr.lower() or "filtered" in stderr.lower():
            return None
        # Try without match-filter as fallback.
        cmd2 = [c for c in cmd if not c.startswith("--match-filter") and c != f"duration<={int(max_duration)}"]
        try:
            proc2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=180)
            if proc2.returncode != 0:
                return None
        except subprocess.TimeoutExpired:
            return None

    # Find the downloaded file.
    for p in sorted(dest_dir.glob(f"{prefix}_*"), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".webm", ".m4v"}:
            return p

    return None


# ── Trim + prepare clip ─────────────────────────────────────────────────────


def get_video_duration(path: str | Path) -> float:
    """Get duration of a video file in seconds.

    Uses ``ffmpeg -i`` (stderr parsing) since imageio-ffmpeg doesn't bundle
    ffprobe on all platforms.  Falls back to ffprobe if available.
    """
    ffmpeg = _ffmpeg_exe()
    try:
        proc = subprocess.run(
            [ffmpeg, "-i", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        # Duration line appears in stderr: "Duration: HH:MM:SS.ff"
        m = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2})\.(\d+)", proc.stderr or "")
        if m:
            h, mi, s, frac = m.groups()
            return int(h) * 3600 + int(mi) * 60 + int(s) + float("0." + frac)
    except Exception:
        pass
    return 0.0


def trim_clip(
    input_path: str | Path,
    output_path: str | Path,
    *,
    start: float = 0.0,
    duration: float = 10.0,
    width: int = 1920,
    height: int = 1080,
    mute: bool = True,
) -> Path:
    """Trim a video clip to the specified duration, scale/crop to target dimensions, optionally mute.

    Returns the output path.
    """

    ffmpeg = _ffmpeg_exe()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}:(in_w-out_w)/2:(in_h-out_h)/2,"
        f"setsar=1,format=yuv420p"
    )

    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start:.3f}",
        "-i", str(input_path),
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-r", "30",
        "-pix_fmt", "yuv420p",
    ]

    if mute:
        cmd += ["-an"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "128k"]

    cmd.append(str(output_path))

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {proc.stderr[:500]}")

    return output_path


def find_best_clip_segment(
    clip_path: str | Path,
    *,
    target_duration: float,
    model: str = "gpt-4o-mini",
) -> tuple[float, float]:
    """Determine the best start offset within a downloaded clip.

    For short clips (< 2x target), just use the beginning.
    For longer clips, pick a segment with likely action/interest.
    Returns (start_seconds, duration_seconds).
    """

    clip_dur = get_video_duration(clip_path)
    if clip_dur <= 0:
        return 0.0, target_duration

    if clip_dur <= target_duration * 1.5:
        # Clip is about the right length; use from start, capped.
        return 0.0, min(clip_dur, target_duration)

    # For longer clips, pick a segment roughly 1/3 to 2/3 in (avoid intros/outros).
    # Simple heuristic: start at ~25% of clip.
    start = max(0.0, clip_dur * 0.25)
    if start + target_duration > clip_dur:
        start = max(0.0, clip_dur - target_duration)

    return start, min(target_duration, clip_dur - start)


# ── High-level: search → download → prepare ─────────────────────────────────


@dataclass
class PreparedClip:
    """A video clip ready to be spliced into the final video."""

    path: Path  # trimmed, scaled, muted clip path
    timeline_start: float  # when in the output video this clip begins
    timeline_end: float  # when it ends
    source_url: str
    source_title: str
    search_query: str
    muted: bool


def prepare_clip_for_suggestion(
    suggestion: "VideoClipSuggestion",
    *,
    dest_dir: str | Path,
    clip_index: int = 0,
    width: int = 1920,
    height: int = 1080,
    llm_model: str = "gpt-4o-mini",
) -> PreparedClip | None:
    """End-to-end: search → pick best → download → trim → return prepared clip.

    Returns None if any step fails (non-fatal).
    """

    from .models import VideoClipSuggestion  # deferred for type safety

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    target_duration = float(suggestion.timeline_end - suggestion.timeline_start)
    if target_duration <= 0:
        return None

    # 1) Search.
    results = search_video_clips(
        suggestion.search_query,
        max_results=5,
        preferred_max_duration=max(120.0, target_duration * 10),
    )
    if not results:
        return None

    # 2) Pick best with LLM.
    best_idx = _pick_best_clip_with_llm(
        results,
        search_query=suggestion.search_query,
        reason=suggestion.reason,
        model=llm_model,
    )
    chosen = results[best_idx]

    # 3) Download.
    raw_dir = dest_dir / "raw"
    raw_path = download_clip(
        chosen.url,
        raw_dir,
        prefix=f"clip{clip_index:02d}",
        max_duration=max(300.0, target_duration * 10),
    )
    if raw_path is None:
        # Try next candidate.
        for fallback_idx, fallback in enumerate(results):
            if fallback_idx == best_idx:
                continue
            raw_path = download_clip(
                fallback.url,
                raw_dir,
                prefix=f"clip{clip_index:02d}_fb{fallback_idx}",
                max_duration=max(300.0, target_duration * 10),
            )
            if raw_path is not None:
                chosen = fallback
                break
        if raw_path is None:
            return None

    # 4) Find best segment within the clip.
    clip_start, clip_dur = find_best_clip_segment(
        raw_path,
        target_duration=target_duration,
        model=llm_model,
    )

    # 5) Trim + scale + optionally mute.
    trimmed_path = dest_dir / f"prepared_{clip_index:02d}.mp4"
    try:
        trim_clip(
            raw_path,
            trimmed_path,
            start=clip_start,
            duration=clip_dur,
            width=width,
            height=height,
            mute=suggestion.mute,
        )
    except Exception:
        return None

    return PreparedClip(
        path=trimmed_path,
        timeline_start=suggestion.timeline_start,
        timeline_end=suggestion.timeline_end,
        source_url=chosen.url,
        source_title=chosen.title,
        search_query=suggestion.search_query,
        muted=suggestion.mute,
    )
