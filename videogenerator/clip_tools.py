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
    from .models import CommentaryClipSuggestion, VideoClipSuggestion

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
    view_count: int | None = None  # approximate views for popularity ranking


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


def _parse_view_count(raw: Any) -> int | None:
    """Parse view count from SerpAPI result (e.g. '1,234,567 views', '1.2M views')."""
    if raw is None:
        return None
    text = str(raw).lower().replace(",", "").strip()
    m = re.match(r"([\d.]+)\s*(k|m|b)?", text)
    if not m:
        return None
    num = float(m.group(1))
    suffix = m.group(2) or ""
    if suffix == "k":
        num *= 1_000
    elif suffix == "m":
        num *= 1_000_000
    elif suffix == "b":
        num *= 1_000_000_000
    return int(num)


def search_video_clips(
    query: str,
    *,
    max_results: int = 5,
    preferred_max_duration: float = 300.0,
    sort_by_views: bool = False,
) -> list[VideoSearchResult]:
    """Search for video clips using SerpAPI YouTube search.

    When *sort_by_views* is True the results are requested sorted by view-count
    so the most popular / viral clips appear first.

    Falls back to Google Video search if YouTube-specific search isn't available.
    """

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        print(f"[clip_search] SERPAPI_API_KEY not set — skipping search for: {query}")
        return []

    import requests

    print(f"[clip_search] Searching YouTube: {query!r}  (sort_by_views={sort_by_views})")
    results: list[VideoSearchResult] = []

    # Strategy 1: SerpAPI YouTube search.
    try:
        params = {
            "engine": "youtube",
            "search_query": query,
            "api_key": api_key,
        }
        # sp=CAMSAhAB → sort by view count (most popular first)
        if sort_by_views:
            params["sp"] = "CAMSAhAB"
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

            # Parse view count for popularity ranking.
            views = _parse_view_count(item.get("views"))

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
                    view_count=views,
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

    print(f"[clip_search] Found {len(results)} results for {query!r}")
    for r in results[:5]:
        views_str = f" ({r.view_count:,} views)" if r.view_count else ""
        print(f"  - {r.title[:80]}{views_str}  [{r.url}]")

    return results


def _pick_best_clip_with_llm(
    candidates: list[VideoSearchResult],
    *,
    search_query: str,
    reason: str,
    model: str = "gpt-4o-mini",
    mode: str = "review",
) -> int:
    """Use the LLM to pick the best clip from search results. Returns index.

    *mode* controls the selection criteria:
    - ``"review"`` – prefer official trailers, movie scene clips, behind-the-scenes.
    - ``"commentary"`` – prefer the **most popular / viral** clip (highest views,
      from major news outlets or well-known channels).
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not candidates:
        return 0

    import requests

    compact = []
    for i, c in enumerate(candidates[:10]):
        entry: dict[str, Any] = {
            "i": i,
            "title": c.title[:120],
            "url": c.url,
            "source": c.source,
            "duration": c.duration_seconds,
        }
        if c.view_count is not None:
            entry["views"] = c.view_count
        compact.append(entry)

    if mode == "commentary":
        system = (
            "You are selecting the best video clip for a YouTube **commentary** video.\n"
            "The goal is to find the clip where the SPECIFIC PERSON named in the search query "
            "is actually speaking, reacting, or being featured.\n\n"
            "Selection criteria (in STRICT priority order):\n"
            "1. **The specific person MUST be in the title** — if the search query says "
            "'Megyn Kelly', the clip title MUST mention Megyn Kelly. Do NOT pick a clip "
            "about a different person.\n"
            "2. **Reaction/response clips** — prefer clips where the person is REACTING, "
            "RESPONDING, or RANTING about the topic (words like 'reacts', 'responds', "
            "'slams', 'blasts', 'rant', 'goes off').\n"
            "3. **Post-event over pre-event** — prefer clips recorded AFTER the event "
            "happened (reactions) over clips from BEFORE (previews, plans, announcements).\n"
            "4. **Major news outlets** (Fox News, CNN, MSNBC, BBC, NBC, Daily Wire, etc.)\n"
            "5. **View count** — higher views preferred, but relevance beats popularity.\n"
            "6. **Appropriate length** — 30 seconds to 10 minutes.\n\n"
            "CRITICAL: If the search query names a specific person (e.g., 'Megyn Kelly', "
            "'Ben Shapiro', 'Trump') you MUST pick a clip that features THAT person. "
            "Never pick a clip about a different person just because it has more views.\n\n"
            "Avoid: music videos, full movies, unrelated content, performance clips.\n"
            "Return ONLY valid JSON: {\"index\": <int>}"
        )
    else:
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


def _ensure_ffmpeg_shim() -> str:
    """Ensure yt-dlp can find ffmpeg. Returns the ffmpeg directory."""
    ffmpeg_path = Path(_ffmpeg_exe())
    ffmpeg_dir = str(ffmpeg_path.parent)
    expected_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    shim = ffmpeg_path.parent / expected_name
    if not shim.exists():
        try:
            shim.symlink_to(ffmpeg_path)
        except OSError:
            import shutil as _shutil
            _shutil.copy2(ffmpeg_path, shim)
    return ffmpeg_dir


def get_video_info(url: str) -> dict[str, Any] | None:
    """Get video metadata (title, duration) without downloading.

    Returns dict with 'title', 'duration' (seconds), or None on failure.
    """
    ytdlp = _find_ytdlp()
    if not ytdlp:
        return None

    try:
        cmd = [
            ytdlp,
            "--no-playlist",
            "--print", "%(title)s\n%(duration)s",
            "--no-download",
            "--no-warnings",
            url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            return None
        lines = proc.stdout.strip().split("\n")
        if len(lines) >= 2:
            title = lines[0].strip()
            try:
                duration = float(lines[1].strip())
            except (ValueError, TypeError):
                duration = 0.0
            return {"title": title, "duration": duration}
    except Exception:
        pass
    return None


def download_clip_section(
    url: str,
    dest_dir: str | Path,
    *,
    start: float = 0.0,
    end: float = 30.0,
    prefix: str = "clip",
) -> Path | None:
    """Download only a specific section of a video using yt-dlp.

    Uses ``--download-sections`` to avoid downloading the entire video.
    Works for any video length — even multi-hour podcasts.
    Returns the downloaded file path, or None on failure.
    """
    ytdlp = _find_ytdlp()
    if not ytdlp:
        raise RuntimeError("yt-dlp is not installed.")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_dir = _ensure_ffmpeg_shim()

    out_template = str(dest_dir / f"{prefix}_%(id)s.%(ext)s")

    # --download-sections "*START-END" tells yt-dlp to download only that range.
    # Add 5s padding on each side for better keyframe alignment.
    padded_start = max(0.0, start - 5.0)
    padded_end = end + 5.0
    section_spec = f"*{padded_start:.1f}-{padded_end:.1f}"

    cmd = [
        ytdlp,
        "--no-playlist",
        "--ffmpeg-location", ffmpeg_dir,
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        "--socket-timeout", "30",
        "--retries", "3",
        "--no-overwrites",
        "--no-post-overwrites",
        "--download-sections", section_spec,
        "--force-keyframes-at-cuts",
        url,
    ]

    print(f"[download] Downloading section {padded_start:.0f}s-{padded_end:.0f}s from {url}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"[download] Timeout downloading section")
        return None

    if proc.returncode != 0:
        stderr = proc.stderr or ""
        print(f"[download] Section download failed: {stderr[:200]}")
        # Fallback: try without --download-sections (full download) with match-filter
        cmd_fallback = [
            ytdlp,
            "--no-playlist",
            "--ffmpeg-location", ffmpeg_dir,
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]/best",
            "--merge-output-format", "mp4",
            "-o", out_template,
            "--max-filesize", "200M",
            "--socket-timeout", "30",
            "--retries", "2",
            "--no-overwrites",
            "--no-post-overwrites",
            url,
        ]
        try:
            proc2 = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=600)
            if proc2.returncode != 0:
                return None
        except subprocess.TimeoutExpired:
            return None

    # Find the downloaded file.
    for p in sorted(dest_dir.glob(f"{prefix}_*"), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".webm", ".m4v"}:
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

    # Point yt-dlp at the imageio-ffmpeg bundled binary.
    ffmpeg_dir = _ensure_ffmpeg_shim()

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
    search_query: str = "",
    video_title: str = "",
) -> tuple[float, float]:
    """Determine the best start offset within a downloaded clip.

    For short clips (< 2x target), just use the beginning.
    For longer clips, use the LLM with the video title and search context
    to pick the most relevant segment.
    Returns (start_seconds, duration_seconds).
    """

    clip_dur = get_video_duration(clip_path)
    if clip_dur <= 0:
        return 0.0, target_duration

    if clip_dur <= target_duration * 1.5:
        # Clip is about the right length; use from start, capped.
        return 0.0, min(clip_dur, target_duration)

    # Use LLM to pick the best segment when we have context.
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and (search_query or video_title):
        try:
            import requests as _req
            system = (
                "You are helping select the best segment from a YouTube video to use as a clip.\n"
                "Given the video title, total duration, what we searched for, and how long the clip should be,\n"
                "pick the best START time (in seconds) so the clip shows the most relevant/interesting part.\n\n"
                "Guidelines:\n"
                "- Skip intros, outros, channel branding (usually first 5-15s and last 10s).\n"
                "- For news clips: jump to where the person of interest is actually speaking.\n"
                "- For reaction videos: jump to the peak reaction moment.\n"
                "- For interviews: jump to the key quote or heated exchange.\n"
                "- Make sure start + duration doesn't exceed the total video duration.\n\n"
                'Return ONLY valid JSON: {"start": <float>}'
            )
            user_msg = {
                "video_title": video_title,
                "video_duration_seconds": round(clip_dur, 1),
                "search_query": search_query,
                "desired_clip_duration": round(target_duration, 1),
            }
            resp = _req.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)},
                ], "temperature": 0.2, "max_tokens": 60},
                timeout=20,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(content)
            start = float(parsed.get("start", 0))
            start = max(0.0, min(start, clip_dur - target_duration))
            dur = min(target_duration, clip_dur - start)
            print(f"[clip_segment] LLM picked start={start:.1f}s for '{video_title[:60]}' (total {clip_dur:.1f}s)")
            return start, dur
        except Exception:
            pass  # fall back to heuristic

    # Fallback: pick a segment roughly 1/3 in (avoid intros/outros).
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
    sort_by_views: bool = False,
    mode: str = "review",
    seen_urls: set[str] | None = None,
) -> PreparedClip | None:
    """End-to-end: search → pick best → download → trim → return prepared clip.

    When *sort_by_views* is True, YouTube results are sorted by view count
    so the most popular / viral clip is preferred.
    *mode* is passed to the LLM clip picker (``"review"`` or ``"commentary"``).

    Returns None if any step fails (non-fatal).
    """

    from .models import VideoClipSuggestion  # deferred for type safety

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    target_duration = float(suggestion.timeline_end - suggestion.timeline_start)
    if target_duration <= 0:
        return None

    # 1) Search — try multiple query variations for commentary mode to find the right clip.
    max_dur = max(120.0, target_duration * 10)
    results = search_video_clips(
        suggestion.search_query,
        max_results=8,
        preferred_max_duration=max_dur,
        sort_by_views=sort_by_views,
    )
    if mode == "commentary":
        # Also search with reaction-focused variations to find actual reaction clips.
        base_q = suggestion.search_query
        seen_result_urls = {r.url for r in results}
        for suffix in [" reacts", " reaction", " responds", " rant"]:
            if suffix.strip() in base_q.lower():
                continue  # already contains this word
            extra = search_video_clips(
                base_q + suffix,
                max_results=5,
                preferred_max_duration=max_dur,
                sort_by_views=sort_by_views,
            )
            for r in extra:
                if r.url not in seen_result_urls:
                    results.append(r)
                    seen_result_urls.add(r.url)
    if not results:
        return None

    # Filter out already-used URLs to avoid duplicates across clips.
    if seen_urls:
        results = [r for r in results if r.url not in seen_urls] or results[:1]

    # 2) Pick best with LLM.
    best_idx = _pick_best_clip_with_llm(
        results,
        search_query=suggestion.search_query,
        reason=suggestion.reason,
        model=llm_model,
        mode=mode,
    )
    chosen = results[best_idx]
    if seen_urls is not None:
        seen_urls.add(chosen.url)

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
        search_query=suggestion.search_query,
        video_title=chosen.title,
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


# ── Compilation: download multiple clips → montage ──────────────────────────


def _concat_clips_ffmpeg(
    clip_paths: list[Path],
    output_path: Path,
    *,
    width: int = 1920,
    height: int = 1080,
    mute: bool = True,
) -> Path:
    """Concatenate multiple trimmed clips into a single montage video."""

    ffmpeg = _ffmpeg_exe()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a concat demuxer file.
    concat_file = output_path.parent / f"{output_path.stem}_concat.txt"
    lines: list[str] = []
    for p in clip_paths:
        safe = str(p.resolve()).replace("\\", "/")
        lines.append(f"file '{safe}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [
        ffmpeg, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
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

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {(proc.stderr or '')[:500]}")
    return output_path


def prepare_compilation_clip(
    suggestion: "CommentaryClipSuggestion",
    *,
    dest_dir: str | Path,
    clip_index: int = 0,
    width: int = 1920,
    height: int = 1080,
    llm_model: str = "gpt-4o-mini",
    sort_by_views: bool = False,
    mode: str = "commentary",
) -> PreparedClip | None:
    """Search, download, trim, and concatenate multiple clips into a montage.

    Used for ``compilation``-type commentary clip suggestions.
    Returns a single :class:`PreparedClip` with the concatenated montage, or None.
    """

    from .models import CommentaryClipSuggestion  # deferred

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = dest_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    total_duration = float(suggestion.timeline_end - suggestion.timeline_start)
    if total_duration <= 0:
        return None

    num_clips = max(2, min(6, suggestion.num_clips))
    per_clip_dur = max(2.0, total_duration / num_clips)

    # Gather all search queries (main + extras).
    queries: list[str] = [suggestion.search_query]
    if suggestion.extra_queries:
        queries.extend(suggestion.extra_queries)
    # Pad to num_clips by repeating main query with variations.
    while len(queries) < num_clips:
        queries.append(suggestion.search_query)

    print(f"[compilation] Building montage: {num_clips} clips, {per_clip_dur:.1f}s each, {total_duration:.1f}s total")
    print(f"[compilation] Queries: {queries}")

    trimmed_parts: list[Path] = []
    source_urls: list[str] = []
    source_titles: list[str] = []

    seen_urls: set[str] = set()  # deduplicate across queries

    for qi, query in enumerate(queries[:num_clips]):
        try:
            results = search_video_clips(
                query,
                max_results=8,
                preferred_max_duration=300.0,
                sort_by_views=sort_by_views,
            )
            if not results:
                continue

            # Pick best with LLM.
            best_idx = _pick_best_clip_with_llm(
                results,
                search_query=query,
                reason=suggestion.reason,
                model=llm_model,
                mode=mode,
            )

            # Find first non-duplicate.
            chosen = None
            for offset in range(len(results)):
                candidate = results[(best_idx + offset) % len(results)]
                if candidate.url not in seen_urls:
                    chosen = candidate
                    break
            if chosen is None:
                continue

            seen_urls.add(chosen.url)

            # Download.
            raw_path = download_clip(
                chosen.url,
                raw_dir,
                prefix=f"comp{clip_index:02d}_{qi:02d}",
                max_duration=300.0,
            )
            if raw_path is None:
                continue

            # Find segment + trim.
            clip_start, clip_dur = find_best_clip_segment(
                raw_path,
                target_duration=per_clip_dur,
                model=llm_model,
                search_query=query,
                video_title=chosen.title,
            )

            part_path = dest_dir / f"comp{clip_index:02d}_part{qi:02d}.mp4"
            trim_clip(
                raw_path,
                part_path,
                start=clip_start,
                duration=clip_dur,
                width=width,
                height=height,
                mute=suggestion.mute,
            )
            trimmed_parts.append(part_path)
            source_urls.append(chosen.url)
            source_titles.append(chosen.title)

        except Exception:
            continue  # non-fatal

    if not trimmed_parts:
        return None

    # Concatenate all parts into one montage.
    montage_path = dest_dir / f"prepared_{clip_index:02d}.mp4"
    if len(trimmed_parts) == 1:
        # Single clip, just rename / copy.
        import shutil as _shutil
        _shutil.copy2(trimmed_parts[0], montage_path)
    else:
        _concat_clips_ffmpeg(
            trimmed_parts,
            montage_path,
            width=width,
            height=height,
            mute=suggestion.mute,
        )

    actual_dur = get_video_duration(montage_path)

    return PreparedClip(
        path=montage_path,
        timeline_start=suggestion.timeline_start,
        timeline_end=suggestion.timeline_start + (actual_dur if actual_dur > 0 else total_duration),
        source_url=", ".join(source_urls),
        source_title=" | ".join(source_titles),
        search_query=suggestion.search_query,
        muted=suggestion.mute,
    )
