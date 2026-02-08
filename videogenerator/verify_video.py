from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
import re

import imageio_ffmpeg
import numpy as np


@dataclass(frozen=True)
class VideoVerification:
    duration_line: str
    has_audio: bool
    audio_peak: int | None
    frame_hashes: list[str]


def _ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_streams(video_path: str | Path) -> str:
    p = Path(video_path).resolve()
    cmd = [_ffmpeg(), "-hide_banner", "-i", str(p)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stderr


def extract_frames(video_path: str | Path, times_s: list[float], out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []

    for i, t in enumerate(times_s):
        out = out_dir / f"frame_{i:02d}_{t:.2f}.png"
        cmd = [
            _ffmpeg(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(Path(video_path).resolve()),
            # Accurate seeking: place -ss after -i so we don't just land on the first keyframe.
            "-ss",
            str(float(t)),
            "-frames:v",
            "1",
            str(out),
        ]
        try:
            subprocess.run(cmd, check=False)
        except Exception:
            continue
        if out.exists():
            frames.append(out)

    return frames


def _parse_duration_seconds(stderr: str) -> float | None:
    # Typical line: Duration: 00:01:23.45, start: 0.000000, bitrate: ...
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = float(m.group(3))
    return hh * 3600.0 + mm * 60.0 + ss


def audio_peak_first_seconds(video_path: str | Path, seconds: float = 3.0) -> int | None:
    cmd = [
        _ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(Path(video_path).resolve()),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "-t",
        str(float(seconds)),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    audio = np.frombuffer(proc.stdout, np.int16)
    if audio.size == 0:
        return None
    return int(np.max(np.abs(audio)))


def verify_local(video_path: str | Path) -> VideoVerification:
    stderr = probe_streams(video_path)
    duration_line = next((ln for ln in stderr.splitlines() if "Duration:" in ln), "")
    has_audio = any("Audio:" in ln for ln in stderr.splitlines())
    peak = audio_peak_first_seconds(video_path, seconds=3.0) if has_audio else None

    # frame hashes: use raw bytes hash to detect if frames are identical
    import hashlib

    tmp = Path(video_path).with_suffix("")
    frame_dir = tmp.parent / (tmp.name + "_frames")

    dur_s = _parse_duration_seconds(stderr)
    if dur_s and dur_s > 0.5:
        # Sample across the video duration, avoiding the exact end.
        end = max(0.0, float(dur_s) - 0.25)
        times = [0.0, end * 0.25, end * 0.50, end * 0.75, end]
    else:
        # Fallback to a conservative set.
        times = [0.0, 2.0, 5.0, 8.0, 10.0]

    frames = extract_frames(video_path, times, frame_dir)
    hashes: list[str] = []
    for f in frames:
        try:
            h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            hashes.append(h)
        except FileNotFoundError:
            continue

    return VideoVerification(duration_line=duration_line, has_audio=has_audio, audio_peak=peak, frame_hashes=hashes)
