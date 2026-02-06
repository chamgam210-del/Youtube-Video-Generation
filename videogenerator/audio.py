from __future__ import annotations

from pathlib import Path

from mutagen import File as MutagenFile


def get_audio_duration_seconds(audio_path: str | Path) -> float:
    audio_path = str(audio_path)
    mf = MutagenFile(audio_path)
    if mf is None or not hasattr(mf, "info") or mf.info is None or not hasattr(mf.info, "length"):
        raise RuntimeError(f"Unable to read audio duration: {audio_path}")
    return float(mf.info.length)
