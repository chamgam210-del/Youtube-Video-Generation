from __future__ import annotations

from pathlib import Path

import imageio_ffmpeg
import numpy as np
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone

from .models import TranscriptSegment
from .utils import ensure_dir, read_json, sanitize_filename, write_json


def _audio_content_hash(audio_path: Path) -> str:
    """Fast content-based hash using file size + first/last 64 KB.

    This avoids re-transcribing when only the file's mtime changes
    (e.g. after a copy or re-save without content changes).
    """
    import hashlib

    h = hashlib.sha256()
    size = audio_path.stat().st_size
    h.update(size.to_bytes(8, "little"))
    chunk = 65536
    with open(audio_path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk * 2:
            f.seek(-chunk, 2)
        h.update(f.read(chunk))
    return h.hexdigest()[:24]


def _decode_audio_to_float32_mono_16k(audio_path: str | Path) -> np.ndarray:
    """Decode audio file to a 16kHz mono float32 waveform in [-1, 1].

    Uses an absolute ffmpeg binary from imageio-ffmpeg so it works on Windows
    without a system ffmpeg installation.
    """

    ffmpeg = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve())
    cmd = [
        ffmpeg,
        "-nostdin",
        "-i",
        str(Path(audio_path).resolve()),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-",
    ]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg decode failed ({proc.returncode}):\n{stderr}")

    audio = np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0
    return audio


def transcribe_with_whisper(audio_path: str | Path, model_name: str = "small") -> tuple[list[TranscriptSegment], list[dict]]:
    """Transcribe using local Whisper (openai-whisper).

    Returns (segments, words) where *words* contains per-word timestamps.
    """

    try:
        import whisper  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Local transcription requires 'openai-whisper'. Install it with: pip install openai-whisper"
        ) from e

    model = whisper.load_model(model_name)

    audio = _decode_audio_to_float32_mono_16k(audio_path)
    result = model.transcribe(audio, fp16=False, word_timestamps=True)
    segments: list[TranscriptSegment] = []
    all_words: list[dict] = []

    for seg in result.get("segments", []) or []:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if end <= start:
            continue
        segments.append(TranscriptSegment(start=start, end=end, text=text))
        # Collect word-level timestamps.
        for w in seg.get("words", []) or []:
            all_words.append({
                "word": str(w.get("word", "")),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
            })

    if not segments:
        text = str(result.get("text", "")).strip()
        if text:
            # Fallback: single segment with unknown end (caller may expand)
            segments.append(TranscriptSegment(start=0.0, end=0.0, text=text))

    return segments, all_words


def transcribe_cached(
    audio_path: str | Path,
    *,
    model_name: str = "small",
    cache_dir: str | Path = ".cache/transcripts",
    use_cache: bool = True,
) -> list[TranscriptSegment]:
    audio_path = Path(audio_path)
    cache_dir = ensure_dir(cache_dir)

    st = audio_path.stat()
    content_hash = _audio_content_hash(audio_path)
    cache_key = sanitize_filename(f"{audio_path.stem}.{model_name}")
    cache_path = cache_dir / f"{cache_key}.json"

    if use_cache and cache_path.exists():
        try:
            data = read_json(cache_path)
            meta = data.get("meta", {}) if isinstance(data, dict) else {}
            # Match by content hash (preferred) or legacy mtime+size.
            cached_hash = meta.get("content_hash", "")
            hash_match = cached_hash and cached_hash == content_hash
            legacy_match = (
                not cached_hash
                and meta.get("audio_name") == audio_path.name
                and int(meta.get("audio_size", -1)) == int(st.st_size)
                and float(meta.get("audio_mtime", -1)) == float(st.st_mtime)
            )
            if (
                (hash_match or legacy_match)
                and meta.get("model") == model_name
            ):
                segs = data.get("segments") or []
                out: list[TranscriptSegment] = []
                for s in segs:
                    out.append(
                        TranscriptSegment(
                            start=float(s["start"]),
                            end=float(s["end"]),
                            text=str(s["text"]),
                        )
                    )
                if out:
                    # Back-fill content_hash into legacy cache entries.
                    if not cached_hash:
                        meta["content_hash"] = content_hash
                        try:
                            write_json(cache_path, data)
                        except Exception:
                            pass
                    return out
        except Exception:
            # Cache read/parse errors fall back to fresh transcription.
            pass

    segments, words = transcribe_with_whisper(audio_path, model_name=model_name)

    if use_cache:
        payload = {
            "meta": {
                "audio_name": audio_path.name,
                "audio_size": int(st.st_size),
                "audio_mtime": float(st.st_mtime),
                "content_hash": content_hash,
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "segments": [asdict(s) for s in segments],
            "words": words,
        }
        write_json(cache_path, payload)

    return segments


def write_transcript_files(
    segments: list[TranscriptSegment],
    *,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    out_dir = ensure_dir(out_dir)

    json_path = out_dir / "transcript.json"
    txt_path = out_dir / "transcript.txt"

    write_json(json_path, [asdict(s) for s in segments])

    lines: list[str] = []
    for s in segments:
        lines.append(f"[{s.start:7.2f} - {s.end:7.2f}] {s.text}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return json_path, txt_path


def get_transcript_cache_path(
    audio_path: str | Path,
    *,
    model_name: str = "small",
    cache_dir: str | Path = ".cache/transcripts",
) -> Path:
    """Return the cache JSON path for a given audio file (may or may not exist)."""
    audio_path = Path(audio_path)
    cache_dir = Path(cache_dir)
    cache_key = sanitize_filename(f"{audio_path.stem}.{model_name}")
    return cache_dir / f"{cache_key}.json"
