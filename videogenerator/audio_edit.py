from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path

import imageio_ffmpeg

from .llm_storyboard import HighlightClip


def _ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def concat_audio_clips_to_wav(
    audio_path: str | Path,
    clips: list[HighlightClip],
    *,
    out_wav: str | Path,
    sample_rate: int = 44100,
) -> Path:
    """Concatenate multiple timestamped audio windows into a single WAV.

    Uses a single ffmpeg filter_complex with atrim + concat.
    """

    audio_path = Path(audio_path)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    if not clips:
        raise ValueError("No clips provided")

    parts: list[str] = []
    labels: list[str] = []

    for i, c in enumerate(clips):
        start = max(0.0, float(c.start))
        end = max(start + 0.01, float(c.end))
        lbl = f"a{i}"
        labels.append(f"[{lbl}]")
        parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS,aresample={int(sample_rate)}[{lbl}]"
        )

    concat = "".join(labels) + f"concat=n={len(labels)}:v=0:a=1[aout]"
    filter_complex = ";".join(parts + [concat])

    cmd = [
        _ffmpeg(),
        "-y",
        "-hide_banner",
        "-i",
        str(audio_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[aout]",
        "-ac",
        "2",
        "-ar",
        str(int(sample_rate)),
        str(out_wav),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg audio concat failed ({}):\n{}".format(proc.returncode, (proc.stderr or "").strip()[:4000])
        )

    return out_wav


def write_highlight_clips_json(out_path: str | Path, clips: list[HighlightClip]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    out_path.write_text(json.dumps([asdict(c) for c in clips], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
