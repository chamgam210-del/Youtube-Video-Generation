"""Create a YouTube Short (≤60s, 9:16) from a full-length video.

Takes a continuous chunk from the start of the video (after skipping any
intro branding), crops to 9:16, speeds it up, and caps at the desired
output duration.

Usage
-----
    uv run python -m videogenerator.create_short \\
        --video output/video.mp4 \\
        --out output/short.mp4 \\
        [--output-duration 55] [--speed 1.35] [--skip-intro 2.5]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd: list[str], label: str = "") -> subprocess.CompletedProcess[str]:
    print(f"  [ffmpeg] {label or ' '.join(cmd[:6])}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"  stderr: {r.stderr[-600:]}", file=sys.stderr, flush=True)
        raise RuntimeError(f"ffmpeg failed ({label})")
    return r


def _video_duration(path: str) -> float:
    ffmpeg = _ffmpeg()
    r = subprocess.run(
        [ffmpeg, "-i", path, "-hide_banner"],
        capture_output=True, text=True, timeout=15,
    )
    for line in (r.stderr or "").splitlines():
        if "Duration:" in line:
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", line)
            if m:
                return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    raise RuntimeError(f"Cannot determine duration of {path}")


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

def create_youtube_short(
    *,
    video_path: str,
    out_path: str = "output/short.mp4",
    output_duration: float = 55,
    speed: float = 1.35,
    skip_intro: float = 2.5,
) -> Path:
    """Create a YouTube Short from a full-length video.

    Takes one continuous chunk from the beginning of *video_path* (after
    skipping *skip_intro* seconds of branding), crops to 9:16 centre,
    speeds up by *speed*×, and caps the output at *output_duration* seconds.

    Parameters
    ----------
    video_path : str
        Path to the source video (landscape 16:9).
    out_path : str
        Destination for the Short MP4.
    output_duration : float
        Desired length of the final Short in seconds (default 55).
    speed : float
        Playback speed multiplier (default 1.35).
    skip_intro : float
        Seconds to skip at the very start (e.g. channel intro, default 2.5).
    """

    print(f"\n{'='*60}")
    print(f"Creating YouTube Short from: {video_path}")
    print(f"  output_duration={output_duration}s  speed={speed}x  skip_intro={skip_intro}s")
    print(f"{'='*60}\n")

    # Validate source
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Source video not found: {video_path}")

    vid_duration = _video_duration(video_path)
    print(f"Source video duration: {vid_duration:.1f}s", flush=True)

    # Calculate how much source footage we need
    source_duration = output_duration * speed
    start_time = skip_intro
    available = vid_duration - start_time
    if source_duration > available:
        print(f"  ⚠ Requested {source_duration:.1f}s of source but only "
              f"{available:.1f}s available after intro skip; using all.", flush=True)
        source_duration = available

    print(f"  Source chunk: {start_time:.1f}s → {start_time + source_duration:.1f}s "
          f"({source_duration:.1f}s @ {speed}x → ~{source_duration / speed:.1f}s output)",
          flush=True)

    # Build ffmpeg filters
    # Video: crop 9:16 centre → scale to 1080x1920 → speed up
    vf = (
        f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,"
        f"scale=1080:1920:flags=lanczos,"
        f"setpts=PTS/{speed:.4f}"
    )

    # Audio: atempo (chain if speed > 2.0)
    atempo_chain: list[str] = []
    remaining = speed
    while remaining > 2.0:
        atempo_chain.append("atempo=2.0")
        remaining /= 2.0
    atempo_chain.append(f"atempo={remaining:.4f}")
    af = ",".join(atempo_chain)

    # Ensure output directory exists
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = _ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-ss", f"{start_time:.3f}",
        "-t", f"{source_duration:.3f}",
        "-i", video_path,
        "-vf", vf,
        "-af", af,
        "-t", f"{output_duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out),
    ]
    _run(cmd, f"short {output_duration:.0f}s @ {speed}x")

    # Report result
    final_dur = _video_duration(str(out))
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"✓ YouTube Short created: {out_path}")
    print(f"  Duration: {final_dur:.1f}s | Size: {size_mb:.1f}MB | Resolution: 1080x1920")
    print(f"{'='*60}\n")

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a YouTube Short (9:16) from a full-length video"
    )
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--out", default="output/short.mp4",
                        help="Output path (default: output/short.mp4)")
    parser.add_argument("--output-duration", type=float, default=55,
                        help="Desired Short length in seconds (default: 55)")
    parser.add_argument("--speed", type=float, default=1.35,
                        help="Playback speed multiplier (default: 1.35)")
    parser.add_argument("--skip-intro", type=float, default=2.5,
                        help="Seconds to skip at start (default: 2.5)")
    args = parser.parse_args()

    create_youtube_short(
        video_path=args.video,
        out_path=args.out,
        output_duration=args.output_duration,
        speed=args.speed,
        skip_intro=args.skip_intro,
    )


if __name__ == "__main__":
    main()
