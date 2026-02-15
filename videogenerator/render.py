from __future__ import annotations

import math
import subprocess
import shutil
import wave
import os
import re
from pathlib import Path

import imageio_ffmpeg

from .models import Slide


def _is_encoder_error(stderr: str) -> bool:
    s = (stderr or "").lower()
    return any(
        needle in s
        for needle in (
            "unknown encoder",
            "error selecting an encoder",
            "could not open encoder",
            "encoder not found",
            "requested output format",
        )
    )


def _ffmpeg_exe() -> str:
    # imageio-ffmpeg downloads a working ffmpeg binary per-platform
    return imageio_ffmpeg.get_ffmpeg_exe()


def _bgm_preset_cache_dir() -> Path:
    d = Path.cwd() / ".cache" / "bgm_presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _try_find_bgm_preset_audio_file(key: str) -> Path | None:
    """Resolve a preset name to a user-provided audio file (mp3/wav/etc).

    Supported locations (first match wins):
    - $VIDEOGENERATOR_BGM_DIR/<key>.(mp3|wav|m4a|aac|flac|ogg)
    - <repo_root>/assets/bgm/<key>.(mp3|wav|m4a|aac|flac|ogg)
    - <cwd>/assets/bgm/<key>.(mp3|wav|m4a|aac|flac|ogg)
    """

    k = str(key or "").strip().lower()
    if not k:
        return None

    exts = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")

    def norm_stem(p: Path) -> str:
        s = str(p.stem or "").lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s
    candidates: list[Path] = []

    env_dir = os.getenv("VIDEOGENERATOR_BGM_DIR")
    if env_dir:
        base = Path(env_dir).expanduser()
        for ext in exts:
            candidates.append(base / f"{k}{ext}")

    repo_root = Path(__file__).resolve().parent.parent
    for ext in exts:
        candidates.append(repo_root / "assets" / "bgm" / f"{k}{ext}")

    cwd = Path.cwd().resolve()
    for ext in exts:
        candidates.append(cwd / "assets" / "bgm" / f"{k}{ext}")

    for p in candidates:
        try:
            if p.exists() and p.is_file() and p.stat().st_size > 4096:
                return p
        except Exception:
            continue

    # Fallback: scan typical bgm dirs and accept files whose normalized stem contains the preset key.
    # This lets `cylinder_five` match `Cylinder Five - Chris Zabriskie.mp3`.
    scan_dirs: list[Path] = []
    if env_dir:
        scan_dirs.append(Path(env_dir).expanduser())
    scan_dirs.append(repo_root / "assets" / "bgm")
    scan_dirs.append(cwd / "assets" / "bgm")

    for d in scan_dirs:
        try:
            if not d.exists() or not d.is_dir():
                continue
            for ext in exts:
                for p in d.glob(f"*{ext}"):
                    try:
                        if p.stat().st_size <= 4096:
                            continue
                        if k in norm_stem(p):
                            return p
                    except Exception:
                        continue
        except Exception:
            continue
    return None


def _convert_audio_file_to_wav(*, in_path: Path, out_wav: Path, seconds: float, sample_rate: int = 44100) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(in_path.resolve()),
        "-t",
        str(max(1.0, float(seconds))),
        "-ac",
        "2",
        "-ar",
        str(int(sample_rate)),
        "-c:a",
        "pcm_s16le",
        str(out_wav.resolve()),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg audio convert failed: {p.stderr or p.stdout}")


def bgm_preset_available(preset: str) -> bool:
    """Return True if `preset` is a known built-in or a resolvable local file-backed preset."""

    key = str(preset or "").strip().lower()
    if not key:
        return False
    if key in ("ambient", "pad"):
        return True
    if key in (
        "elevator",
        "elevator_music",
        "creepy",
        "horror",
        "spooky",
        "hiphop",
        "hip_hop",
        "hip-hop",
        "hip hop",
        "rnb",
        "r&b",
        "rb",
        "r_b",
        "clown",
        "circus",
        "clown_music",
        "circus_music",
        "mocking",
    ):
        return True
    return _try_find_bgm_preset_audio_file(key) is not None


def ensure_bgm_preset_wav(*, preset: str, seconds: float = 32.0) -> Path:
    """Create (if needed) and return the WAV path for a built-in preset.

    This makes presets tangible .wav files that can be played directly, and lets
    rendering reuse cached audio instead of regenerating every run.
    """

    key = str(preset or "").strip().lower()
    if not key:
        raise ValueError("preset must be non-empty")

    cache = _bgm_preset_cache_dir()
    seconds = max(1.0, float(seconds))

    if key in ("ambient", "pad"):
        # Not a wav file; ambient uses lavfi generation in ffmpeg.
        raise ValueError("ambient preset is generated in ffmpeg (no wav)")

    # File-backed preset: let users drop an mp3/wav into assets/bgm/<preset>.*
    # and use it by name (e.g. bgm_preset='cylinder_five').
    audio_file = _try_find_bgm_preset_audio_file(key)
    if audio_file is not None:
        out = cache / f"bgm_{key}.wav"
        try:
            if out.exists() and out.stat().st_size > 4096:
                # Regenerate if the source is newer.
                try:
                    if audio_file.stat().st_mtime <= out.stat().st_mtime:
                        return out
                except Exception:
                    return out
        except Exception:
            pass

        _convert_audio_file_to_wav(in_path=audio_file, out_wav=out, seconds=seconds)
        return out

    if key in ("elevator", "elevator_music"):
        out = cache / "bgm_elevator.wav"
        gen = _generate_elevator_music_wav
        gen_seconds = seconds
    elif key in ("creepy", "horror", "spooky"):
        out = cache / "bgm_creepy.wav"
        gen = _generate_creepy_music_wav
        gen_seconds = seconds
    elif key in ("hiphop", "hip_hop", "hip-hop", "hip hop"):
        out = cache / "bgm_hiphop.wav"
        gen = _generate_hiphop_music_wav
        gen_seconds = seconds
    elif key in ("rnb", "r&b", "rb", "r_b"):
        out = cache / "bgm_rnb.wav"
        gen = _generate_rnb_music_wav
        gen_seconds = seconds
    elif key in ("clown", "circus", "clown_music", "circus_music", "mocking"):
        out = cache / "bgm_clown.wav"
        gen = _generate_clown_music_wav
        gen_seconds = min(seconds, 24.0)
    else:
        raise ValueError(
            f"Unknown bgm_preset: {preset!r}. "
            "If you meant a local file-backed preset, put the audio in assets/bgm/ "
            "(e.g. 'cylinder_five.mp3' or 'Cylinder Five - Chris Zabriskie.mp3') "
            "or set VIDEOGENERATOR_BGM_DIR."
        )

    try:
        if out.exists() and out.stat().st_size > 4096:
            return out
    except Exception:
        pass

    gen(out_wav=out, seconds=float(gen_seconds))
    return out


def _generate_elevator_music_wav(*, out_wav: Path, seconds: float, sample_rate: int = 44100) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # Simple, loop-friendly jazz-ish bed: 8 bars @ 120bpm -> 16s loop.
    # We synthesize: bass (sine), soft chord pad (few sines), and a light arpeggio.
    bpm = 120.0
    beats_per_bar = 4
    seconds_per_beat = 60.0 / bpm
    bar_seconds = beats_per_bar * seconds_per_beat
    loop_bars = 8
    loop_seconds = loop_bars * bar_seconds

    total_seconds = max(1.0, float(seconds))
    total_frames = int(total_seconds * sample_rate)
    loop_frames = int(loop_seconds * sample_rate)
    if loop_frames <= 0:
        loop_frames = sample_rate

    # Chord progression in C major: Cmaj7, Am7, Dm7, G7 (repeat).
    # Frequencies (Hz) for chord tones (approx):
    chords = [
        [261.63, 329.63, 392.00, 493.88],  # C E G B
        [220.00, 261.63, 329.63, 392.00],  # A C E G
        [293.66, 349.23, 440.00, 523.25],  # D F A C
        [196.00, 246.94, 293.66, 349.23],  # G B D F
    ]
    bass = [
        65.41,  # C2
        55.00,  # A1
        73.42,  # D2
        49.00,  # G1
    ]

    def soft_clip(x: float) -> float:
        # Gentle saturation to keep mix pleasant.
        return math.tanh(x)

    def env(t: float, attack: float, release: float) -> float:
        # One-pole-ish envelope: ramp in then ramp out.
        if t < 0:
            return 0.0
        if t < attack:
            return t / max(attack, 1e-6)
        return math.exp(-(t - attack) / max(release, 1e-6))

    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)

        for frame in range(total_frames):
            # Loop time for musical structure.
            lf = frame % loop_frames
            t = lf / sample_rate

            bar_index = int(t // bar_seconds) % loop_bars
            chord_index = bar_index % 4

            beat_in_bar = (t % bar_seconds) / seconds_per_beat
            beat_index = int(beat_in_bar)
            beat_phase = beat_in_bar - beat_index

            # Bass: quarter-note pluck with decay.
            bass_freq = bass[chord_index]
            bass_amp = 0.18 * env(beat_phase * seconds_per_beat, 0.01, 0.18)
            bass_sig = bass_amp * math.sin(2.0 * math.pi * bass_freq * t)

            # Chord pad: on beats 1 and 3 a bit stronger.
            chord_amp_base = 0.08 if beat_index in (0, 2) else 0.05
            chord_env = env(beat_phase * seconds_per_beat, 0.02, 0.35)
            chord_sig = 0.0
            for f in chords[chord_index]:
                chord_sig += math.sin(2.0 * math.pi * f * t)
            chord_sig *= (chord_amp_base * chord_env) / 4.0

            # Arpeggio: 8th-note pattern, very light.
            eighth = (t / (seconds_per_beat / 2.0))
            arp_step = int(eighth) % 8
            arp_note = chords[chord_index][arp_step % 4]
            arp_phase = (eighth - int(eighth)) * (seconds_per_beat / 2.0)
            arp_amp = 0.035 * env(arp_phase, 0.003, 0.08)
            arp_sig = arp_amp * math.sin(2.0 * math.pi * arp_note * t)

            # Slight stereo width via tiny phase offset in right channel.
            mix_l = bass_sig + chord_sig + arp_sig
            mix_r = bass_sig + chord_sig + (arp_amp * math.sin(2.0 * math.pi * arp_note * (t + 0.002)))

            # Keep it gentle.
            mix_l = soft_clip(mix_l)
            mix_r = soft_clip(mix_r)

            # Convert to int16.
            s_l = int(max(-1.0, min(1.0, mix_l)) * 32767)
            s_r = int(max(-1.0, min(1.0, mix_r)) * 32767)
            wf.writeframesraw(int.to_bytes(s_l, 2, "little", signed=True) + int.to_bytes(s_r, 2, "little", signed=True))

        wf.writeframes(b"")


def _generate_creepy_music_wav(*, out_wav: Path, seconds: float, sample_rate: int = 44100) -> None:
    """Generate a loop-friendly creepy drone bed.

    Intentionally simple + deterministic (no external deps): minor intervals, slight detune,
    slow tremolo, and a tiny bit of filtered noise.
    """

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    total_seconds = max(1.0, float(seconds))

    total_frames = int(total_seconds * sample_rate)

    def soft_clip(x: float) -> float:
        return math.tanh(x)

    # Seeded pseudo-random for deterministic noise.
    state = 1337

    def rnd() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return (state / 0x7FFFFFFF) * 2.0 - 1.0

    # Drone chord: D minor-ish cluster with detune.
    base = 73.42  # D2
    freqs = [
        base,
        base * 1.005,
        base * (6.0 / 5.0),  # minor third
        base * (3.0 / 2.0),  # fifth
        2.0 * base * 1.01,
    ]

    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        # Simple one-pole lowpass for noise.
        lp_l = 0.0
        lp_r = 0.0
        alpha = 0.02

        for frame in range(total_frames):
            t = frame / sample_rate

            # Slow tremolo + subtle swell.
            trem = 0.55 + 0.45 * math.sin(2.0 * math.pi * 0.12 * t)
            swell = 0.70 + 0.30 * math.sin(2.0 * math.pi * 0.03 * t + 1.2)
            amp = 0.16 * trem * swell

            drone = 0.0
            for i, f in enumerate(freqs):
                # Spread phases a bit.
                ph = 0.4 * i
                drone += math.sin(2.0 * math.pi * f * t + ph)
            drone /= float(len(freqs))

            # Add a faint high harmonic for tension.
            hiss_tone = 0.03 * math.sin(2.0 * math.pi * (base * 8.0) * t)

            # Light noise, lowpassed.
            n_l = rnd() * 0.08
            n_r = rnd() * 0.08
            lp_l = lp_l + alpha * (n_l - lp_l)
            lp_r = lp_r + alpha * (n_r - lp_r)

            mix_l = amp * (drone + hiss_tone) + (0.04 * lp_l)
            mix_r = amp * (drone + 0.9 * hiss_tone) + (0.04 * lp_r)

            # Very slight stereo movement.
            mix_r += 0.01 * math.sin(2.0 * math.pi * 0.07 * t)

            mix_l = soft_clip(mix_l)
            mix_r = soft_clip(mix_r)

            s_l = int(max(-1.0, min(1.0, mix_l)) * 32767)
            s_r = int(max(-1.0, min(1.0, mix_r)) * 32767)
            wf.writeframesraw(int.to_bytes(s_l, 2, "little", signed=True) + int.to_bytes(s_r, 2, "little", signed=True))

        wf.writeframes(b"")


def _generate_hiphop_music_wav(*, out_wav: Path, seconds: float, sample_rate: int = 44100) -> None:
    """Generate a loop-friendly hip hop beat bed.

    Deterministic, dependency-free synth: kick + snare + hats + simple bass.
    Designed to be looped via ffmpeg (-stream_loop -1).
    """

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    bpm = 92.0
    seconds_per_beat = 60.0 / bpm
    steps_per_beat = 4  # 16th notes
    seconds_per_step = seconds_per_beat / steps_per_beat

    bars = 4
    beats_per_bar = 4
    loop_seconds = bars * beats_per_bar * seconds_per_beat

    total_seconds = max(1.0, float(seconds))
    total_frames = int(total_seconds * sample_rate)
    loop_frames = max(1, int(loop_seconds * sample_rate))

    def soft_clip(x: float) -> float:
        return math.tanh(x)

    def lcg_noise(n: int) -> float:
        # Deterministic pseudo-noise in [-1, 1].
        x = (1103515245 * (n + 12345) + 12345) & 0x7FFFFFFF
        return (x / 0x7FFFFFFF) * 2.0 - 1.0

    def exp_env(t: float, attack: float, decay: float) -> float:
        if t < 0:
            return 0.0
        if t < attack:
            return t / max(attack, 1e-6)
        return math.exp(-(t - attack) / max(decay, 1e-6))

    # 4 bars of 16th-note steps.
    total_steps = bars * beats_per_bar * steps_per_beat
    # Kick on 1, the "and" of 2, and 4 (classic-ish feel).
    kick_steps = {0, 6, 12, 16, 22, 28, 32, 38, 44, 48, 54, 60}
    # Snare on 2 and 4.
    snare_steps = {8, 24, 40, 56}
    # Hats: 8th notes with some 16th fills.
    hat_steps = set(range(0, total_steps, 2)) | {7, 15, 23, 31, 39, 47, 55, 63}

    # Bass pattern (bar-relative), root-ish notes in Hz (roughly E minor vibe).
    bass_notes = [41.20, 49.00, 55.00, 49.00]  # E1, G1, A1, G1

    # Small chunked writer for performance.
    chunk = bytearray()
    chunk_flush_frames = 4096

    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        for frame in range(total_frames):
            lf = frame % loop_frames
            t = lf / sample_rate

            step = int(t // seconds_per_step) % total_steps
            step_t = t - (step * seconds_per_step)

            mix = 0.0

            # Kick: short pitch-swept sine.
            if step in kick_steps:
                kt = step_t
                kenv = exp_env(kt, 0.003, 0.10)
                # Exponential-ish sweep 90 -> 45 Hz.
                f0, f1 = 90.0, 45.0
                sweep = math.exp(-kt / 0.06)
                kfreq = f1 + (f0 - f1) * sweep
                mix += 0.85 * kenv * math.sin(2.0 * math.pi * kfreq * t)

            # Snare: noise burst + quiet tone.
            if step in snare_steps:
                st = step_t
                senv = exp_env(st, 0.002, 0.08)
                noise = lcg_noise(lf) * 0.6 + lcg_noise(lf * 7 + 13) * 0.4
                # Crude band emphasis: mix some high noise.
                sn = 0.50 * noise
                sn += 0.12 * math.sin(2.0 * math.pi * 185.0 * t)
                mix += 0.55 * senv * sn

            # Hi-hats: very short high-frequency noise clicks.
            if step in hat_steps:
                ht = step_t
                henv = exp_env(ht, 0.001, 0.03)
                n = lcg_noise(lf * 3 + 17)
                # Emphasize high end by multiplying with a fast sine.
                hh = n * math.sin(2.0 * math.pi * 8000.0 * t)
                mix += 0.18 * henv * hh

            # Bass: 8th-note-ish sustained sub.
            bass_step = (step // 2)  # 8th note index
            bass_freq = bass_notes[(bass_step // 4) % len(bass_notes)]
            bass_phase = t % (2.0 * seconds_per_beat)
            benv = exp_env(bass_phase, 0.02, 0.35)
            bass = 0.35 * benv * math.sin(2.0 * math.pi * bass_freq * t)
            mix += bass

            # Gentle saturation and safety (slightly hotter so it is audible under narration).
            mix = soft_clip(mix * 1.20)

            s = int(max(-1.0, min(1.0, mix)) * 32767)
            s_l = s
            # Tiny stereo widening via alternating phase noise.
            s_r = int(max(-1.0, min(1.0, soft_clip((mix + 0.02 * lcg_noise(lf * 11 + 5))))) * 32767)

            chunk += int.to_bytes(s_l, 2, "little", signed=True)
            chunk += int.to_bytes(s_r, 2, "little", signed=True)

            if (frame + 1) % chunk_flush_frames == 0:
                wf.writeframesraw(chunk)
                chunk.clear()

        if chunk:
            wf.writeframesraw(chunk)
            chunk.clear()

        wf.writeframes(b"")


def _generate_rnb_music_wav(*, out_wav: Path, seconds: float, sample_rate: int = 44100) -> None:
    """Generate a loop-friendly R&B/hip hop style bed.

    Original, deterministic synth beat: swung hats, tight kick/snare, 808-ish sub bass,
    and a simple electric-piano style chord progression.
    """

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    bpm = 82.0
    spb = 60.0 / bpm
    steps_per_beat = 4  # 16ths
    sps = spb / steps_per_beat
    swing = 0.18  # delay every other 16th slightly

    bars = 4
    beats_per_bar = 4
    loop_seconds = bars * beats_per_bar * spb

    total_seconds = max(1.0, float(seconds))
    total_frames = int(total_seconds * sample_rate)
    loop_frames = max(1, int(loop_seconds * sample_rate))

    def soft_clip(x: float) -> float:
        return math.tanh(x)

    def lcg(n: int) -> int:
        return (1103515245 * (n + 12345) + 12345) & 0x7FFFFFFF

    def noise(n: int) -> float:
        return (lcg(n) / 0x7FFFFFFF) * 2.0 - 1.0

    def exp_env(t: float, attack: float, decay: float) -> float:
        if t < 0:
            return 0.0
        if t < attack:
            return t / max(attack, 1e-6)
        return math.exp(-(t - attack) / max(decay, 1e-6))

    def hz(midi: int) -> float:
        return 440.0 * (2.0 ** ((midi - 69) / 12.0))

    # Chord progression (4 bars): Am7 | Fmaj7 | Cmaj7 | G6
    chords_midi: list[list[int]] = [
        [57, 60, 64, 67],  # A3 C4 E4 G4
        [53, 57, 60, 64],  # F3 A3 C4 E4
        [48, 52, 55, 59],  # C3 E3 G3 B3
        [55, 59, 62, 64],  # G3 B3 D4 E4 (G6-ish)
    ]

    # Drum patterns on 16ths in 1 bar, repeated.
    steps_per_bar = beats_per_bar * steps_per_beat  # 16
    # A common-ish R&B pocket: kick on 1, (1e), 3, (3a); snare on 2 & 4.
    kick_bar = {0, 3, 8, 14}
    snare_bar = {4, 12}
    hat_bar = set(range(0, steps_per_bar, 2)) | {3, 7, 11, 15}  # 8ths + a little texture

    # Bass follows chord root, with some octave jumps.
    bass_roots = [45, 41, 36, 43]  # A2, F2, C2, G2

    chunk = bytearray()
    chunk_flush_frames = 4096

    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        for frame in range(total_frames):
            lf = frame % loop_frames
            t = lf / sample_rate

            # Determine bar/beat/step.
            bar = int(t // (beats_per_bar * spb)) % bars
            t_in_bar = t - bar * (beats_per_bar * spb)

            step_raw = int(t_in_bar // sps)
            step_in_step = t_in_bar - (step_raw * sps)

            # Apply swing by shifting the "off" 16ths later.
            step = step_raw % steps_per_bar
            is_off = (step % 2) == 1
            swing_shift = (sps * swing) if is_off else 0.0
            step_t = max(0.0, step_in_step - swing_shift)

            mix = 0.0

            # --- Drums ---
            if step in kick_bar:
                kt = step_t
                kenv = exp_env(kt, 0.002, 0.12)
                # 808-ish thump: pitch drop 95 -> 48Hz.
                sweep = math.exp(-kt / 0.07)
                kfreq = 48.0 + (95.0 - 48.0) * sweep
                mix += 0.95 * kenv * math.sin(2.0 * math.pi * kfreq * t)
                # click
                mix += 0.08 * exp_env(kt, 0.001, 0.02) * math.sin(2.0 * math.pi * 2400.0 * t)

            if step in snare_bar:
                st = step_t
                senv = exp_env(st, 0.002, 0.10)
                n = (0.7 * noise(lf * 5 + 11) + 0.3 * noise(lf * 13 + 7))
                # snare body + noise
                body = 0.22 * math.sin(2.0 * math.pi * 190.0 * t)
                mix += 0.55 * senv * (0.55 * n + body)

            if step in hat_bar:
                ht = step_t
                henv = exp_env(ht, 0.001, 0.035)
                n = noise(lf * 3 + 17)
                # bright hat
                hh = n * math.sin(2.0 * math.pi * 9000.0 * t)
                mix += 0.18 * henv * hh

            # --- Keys (electric piano-ish) ---
            # Hit chords on beat 1 and a light stab on beat 3.
            chord_hit = (step == 0) or (step == 8)
            if chord_hit:
                # A slightly longer envelope for the chord stab.
                # Use t-based phase; envelope based on time since this step started.
                pass

            # Compute chord contribution continuously but with stronger onset at step 0/8.
            chord = chords_midi[bar % len(chords_midi)]
            # time since last chord onset within bar (either step 0 or step 8)
            onset_step = 0 if step_raw < 8 else 8
            onset_t = t_in_bar - (onset_step * sps)
            cenv = exp_env(onset_t, 0.008, 0.55)
            # simple EP timbre: fundamental + a couple harmonics.
            chord_sig = 0.0
            for m in chord:
                f = hz(m)
                chord_sig += 0.70 * math.sin(2.0 * math.pi * f * t)
                chord_sig += 0.18 * math.sin(2.0 * math.pi * (2.0 * f) * t)
                chord_sig += 0.08 * math.sin(2.0 * math.pi * (3.0 * f) * t)
            chord_sig /= max(1.0, float(len(chord)))
            mix += 0.18 * cenv * chord_sig

            # --- Bass ---
            # Bass notes on 1, (1a), 3, (3&)
            bass_steps = {0, 2, 8, 10}
            if step in bass_steps:
                bt = step_t
                benv = exp_env(bt, 0.004, 0.22)
                root = bass_roots[bar % len(bass_roots)]
                # occasional octave on the second hit
                if step in {2, 10}:
                    root += 12
                bf = hz(root)
                # 808-ish: sine + tiny 2nd harmonic
                bass = math.sin(2.0 * math.pi * bf * t) + 0.18 * math.sin(2.0 * math.pi * (2.0 * bf) * t)
                mix += 0.48 * benv * bass

            # Glue + safety.
            mix = soft_clip(mix * 1.10)

            s = int(max(-1.0, min(1.0, mix)) * 32767)
            # subtle stereo: tiny phase offset in right + slight noise
            r = soft_clip((mix + 0.015 * noise(lf * 11 + 5)))
            s_r = int(max(-1.0, min(1.0, r)) * 32767)

            chunk += int.to_bytes(s, 2, "little", signed=True)
            chunk += int.to_bytes(s_r, 2, "little", signed=True)

            if (frame + 1) % chunk_flush_frames == 0:
                wf.writeframesraw(chunk)
                chunk.clear()

        if chunk:
            wf.writeframesraw(chunk)
            chunk.clear()

        wf.writeframes(b"")


def _generate_clown_music_wav(*, out_wav: Path, seconds: float, sample_rate: int = 44100) -> None:
    """Generate a loop-friendly clown/circus mocking bed.

    Deterministic synth: calliope-ish square lead, oom-pah bass, and tiny snare/hat ticks.
    Intended to be obviously "circus" without using any copyrighted material.
    """

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    bpm = 152.0
    spb = 60.0 / bpm
    beats_per_bar = 4
    bars = 8
    loop_seconds = bars * beats_per_bar * spb

    total_seconds = max(1.0, float(seconds))
    total_frames = int(total_seconds * sample_rate)
    loop_frames = max(1, int(loop_seconds * sample_rate))

    def soft_clip(x: float) -> float:
        return math.tanh(x)

    def lcg(n: int) -> int:
        return (1103515245 * (n + 12345) + 12345) & 0x7FFFFFFF

    def noise(n: int) -> float:
        return (lcg(n) / 0x7FFFFFFF) * 2.0 - 1.0

    def exp_env(t: float, attack: float, decay: float) -> float:
        if t < 0:
            return 0.0
        if t < attack:
            return t / max(attack, 1e-6)
        return math.exp(-(t - attack) / max(decay, 1e-6))

    def hz(midi: int) -> float:
        return 440.0 * (2.0 ** ((midi - 69) / 12.0))

    # Major-scale-ish melody with goofy chromatic dips.
    # (C-ish center, but intentionally a bit cheeky.)
    melody_midi = [
        72, 74, 76, 77,  # C5 D5 E5 F5
        76, 74, 72, 71,  # E5 D5 C5 B4
        72, 74, 72, 69,  # C5 D5 C5 A4
        71, 72, 71, 67,  # B4 C5 B4 G4
    ]
    # Oom-pah bass in C: root/fifth.
    bass_root = 36  # C2
    bass_fifth = 43  # G2

    chunk = bytearray()
    chunk_flush_frames = 4096

    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        for frame in range(total_frames):
            lf = frame % loop_frames
            t = lf / sample_rate

            bar_t = t % (beats_per_bar * spb)
            beat = int(bar_t // spb)
            beat_t = bar_t - beat * spb

            # 8th notes for melody.
            eighth = int((t / (spb / 2.0)))
            eighth_t = (t % (spb / 2.0))
            note = melody_midi[eighth % len(melody_midi)]
            f = hz(note)

            # Calliope-ish lead: square wave with vibrato + a touch of harmonic.
            vib = 1.0 + 0.010 * math.sin(2.0 * math.pi * 6.0 * t)
            phase = 2.0 * math.pi * (f * vib) * t
            lead_sq = 1.0 if math.sin(phase) >= 0 else -1.0
            lead = 0.55 * lead_sq + 0.18 * math.sin(2.0 * math.pi * (2.0 * f) * t)
            lead *= exp_env(eighth_t, 0.004, 0.12)

            # Oom-pah bass: on beats 1 and 3 root; 2 and 4 fifth (staccato).
            bass_m = bass_root if beat in (0, 2) else bass_fifth
            bf = hz(bass_m)
            bass = math.sin(2.0 * math.pi * bf * t)
            bass *= exp_env(beat_t, 0.002, 0.18)

            # Tiny percussion: snare-ish on 2/4 and hat ticks on 8ths.
            sn = 0.0
            if beat in (1, 3):
                sn_env = exp_env(beat_t, 0.001, 0.06)
                sn = sn_env * noise(lf * 7 + 19)

            hat = 0.0
            hat_env = exp_env(eighth_t, 0.001, 0.03)
            hat = hat_env * noise(lf * 5 + 3) * math.sin(2.0 * math.pi * 9500.0 * t)

            mix = 0.32 * lead + 0.30 * bass + 0.10 * sn + 0.06 * hat

            # Little "slide-whistle" style bend at the end of the loop.
            if (loop_seconds - t) < 0.45:
                tt = max(0.0, loop_seconds - t)
                bend = 600.0 + (2200.0 - 600.0) * (1.0 - (tt / 0.45))
                wh_env = exp_env(0.45 - tt, 0.001, 0.20)
                mix += 0.10 * wh_env * math.sin(2.0 * math.pi * bend * t)

            mix = soft_clip(mix * 1.35)

            s_l = int(max(-1.0, min(1.0, mix)) * 32767)
            # light stereo wiggle
            mix_r = soft_clip(mix + 0.012 * noise(lf * 11 + 5))
            s_r = int(max(-1.0, min(1.0, mix_r)) * 32767)

            chunk += int.to_bytes(s_l, 2, "little", signed=True)
            chunk += int.to_bytes(s_r, 2, "little", signed=True)

            if (frame + 1) % chunk_flush_frames == 0:
                wf.writeframesraw(chunk)
                chunk.clear()

        if chunk:
            wf.writeframesraw(chunk)
            chunk.clear()

        wf.writeframes(b"")


def render_slideshow(
    slides: list[Slide],
    audio_path: str | Path,
    out_mp4: str | Path,
    *,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    bgm_path: str | Path | None = None,
    bgm_volume: float = 0.10,
    bgm_duck: bool = True,
    bgm_generate: bool = False,
    bgm_preset: str | None = None,
    intro_seconds: float = 0.0,
    outro_seconds: float = 0.0,
    transition: str | None = "fade",
    transition_seconds: float = 0.35,
    ken_burns: bool = False,
) -> None:
    if not slides:
        raise ValueError("No slides to render")

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    concat_file = out_mp4.parent / "slides.concat.txt"

    def q(p: Path) -> str:
        # ffmpeg concat expects forward slashes; quote with single quotes
        return str(p.resolve()).replace("\\", "/")

    lines: list[str] = []
    durations: list[float] = []
    for s in slides:
        dur = max(0.05, float(s.end - s.start))
        durations.append(dur)
        lines.append(f"file '{q(Path(s.image_path))}'")
        lines.append(f"duration {dur:.3f}")

    # repeat last file without duration (ffmpeg concat demuxer requirement)
    lines.append(f"file '{q(Path(slides[-1].image_path))}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ffmpeg = _ffmpeg_exe()

    if bgm_preset is not None:
        if bgm_path or bgm_generate:
            raise ValueError("Provide only one of bgm_path, bgm_generate, or bgm_preset")
        preset = str(bgm_preset).strip().lower()
        if preset in ("ambient", "pad"):
            bgm_generate = True
        else:
            # Ensure the preset WAV exists (cache) and use it.
            bgm_wav = ensure_bgm_preset_wav(preset=preset, seconds=32.0)
            bgm_path = bgm_wav

            # Also copy into the output folder so it is easy to find/play.
            try:
                local = out_mp4.parent / bgm_wav.name
                if not local.exists():
                    shutil.copyfile(bgm_wav, local)
            except Exception:
                pass

    if bgm_path and bgm_generate:
        raise ValueError("Provide either bgm_path or bgm_generate, not both")

    intro_s = max(0.0, float(intro_seconds))
    outro_s = max(0.0, float(outro_seconds))
    total_duration = float(sum(durations))

    trans = (transition or "").strip().lower() if transition is not None else None
    trans_s = max(0.0, float(transition_seconds))
    if trans is not None and trans not in {"fade"}:
        raise ValueError(f"Unknown transition: {transition!r}")

    intro_ms = int(round(intro_s * 1000.0))
    # adelay expects per-channel delays; provide 2 channels to be safe.
    adelay = f"adelay={intro_ms}|{intro_ms}," if intro_ms > 0 else ""

    def _audio_filters(*, use_duck: bool, voice_label: str, bgm_label: str | None) -> tuple[str | None, str]:
        """Returns (filter_complex_fragment_or_none, audio_map_label)."""
        needs_audio_filter = bool(bgm_label or adelay or outro_s > 0.0)
        if not needs_audio_filter:
            return None, voice_label

        if bgm_label:
            bgm_vol = max(0.0, float(bgm_volume))
            if use_duck:
                frag = (
                    f"[{voice_label}]aformat=sample_fmts=fltp:sample_rates=44100,{adelay}apad,atrim=duration={total_duration:.3f}[voice_in];"
                    "[voice_in]asplit=2[voice_mix][voice_sc];"
                    f"[{bgm_label}]aformat=sample_fmts=fltp:sample_rates=44100,lowpass=f=16000,volume={bgm_vol:.4f},apad,atrim=duration={total_duration:.3f}[bgm];"
                    "[bgm][voice_sc]sidechaincompress=threshold=0.02:ratio=12:attack=5:release=400[bgmduck];"
                    "[voice_mix][bgmduck]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,alimiter=limit=0.97[aout]"
                )
            else:
                frag = (
                    f"[{voice_label}]aformat=sample_fmts=fltp:sample_rates=44100,{adelay}apad,atrim=duration={total_duration:.3f}[voice];"
                    f"[{bgm_label}]aformat=sample_fmts=fltp:sample_rates=44100,lowpass=f=16000,volume={bgm_vol:.4f},apad,atrim=duration={total_duration:.3f}[bgm];"
                    "[voice][bgm]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,alimiter=limit=0.97[aout]"
                )
            return frag, "aout"

        # No BGM, but intro/outro adjustments requested.
        frag = f"[{voice_label}]aformat=sample_fmts=fltp:sample_rates=44100,{adelay}apad,atrim=duration={total_duration:.3f},alimiter=limit=0.97[aout]"
        return frag, "aout"


    def _build_base(*, use_duck: bool) -> list[str]:
        cmd: list[str] = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-i",
            str(Path(audio_path).resolve()),
        ]

        needs_audio_filter = bool(bgm_path or bgm_generate or adelay or outro_s > 0.0)

        if bgm_path or bgm_generate:
            if bgm_path:
                cmd += [
                    "-stream_loop",
                    "-1",
                    "-i",
                    str(Path(bgm_path).resolve()),
                ]
            else:
                cmd += [
                    "-f",
                    "lavfi",
                    "-i",
                    "aevalsrc=0.25*sin(2*PI*110*t)+0.22*sin(2*PI*138.59*t)+0.20*sin(2*PI*164.81*t)+0.18*sin(2*PI*220*t):s=44100",
                ]

            filter_complex, audio_map = _audio_filters(use_duck=use_duck, voice_label="1:a", bgm_label="2:a")
            cmd += [
                "-filter_complex",
                filter_complex or "",
                "-map",
                "0:v:0",
                "-map",
                (f"[{audio_map}]" if audio_map.startswith("a") else audio_map),
            ]
        elif needs_audio_filter:
            filter_complex, audio_map = _audio_filters(use_duck=use_duck, voice_label="1:a", bgm_label=None)
            cmd += [
                "-filter_complex",
                filter_complex or "",
                "-map",
                "0:v:0",
                "-map",
                (f"[{audio_map}]" if audio_map.startswith("a") else audio_map),
            ]
        else:
            cmd += [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]

        # Output options (must come after all inputs).
        cmd += [
            "-fps_mode",
            "cfr",
            "-vf",
            (
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height}:(in_w-out_w)/2:(in_h-out_h)/2,"
                "setsar=1,format=yuv420p"
            ),
            "-r",
            str(fps),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
        ]

        return cmd

    base = _build_base(use_duck=bgm_duck)

    def _build_with_filters(*, use_duck: bool) -> list[str]:
        """Build an ffmpeg command that applies per-slide transitions/effects."""

        img_paths = [Path(s.image_path).resolve() for s in slides]
        voice_path = Path(audio_path).resolve()

        cmd: list[str] = [ffmpeg, "-y"]

        # Track which slides are video clips vs still images.
        slide_is_clip: list[bool] = []
        for i, s in enumerate(slides):
            if s.video_clip_path and Path(s.video_clip_path).exists():
                # Video clip input — no -loop.
                cmd += ["-i", str(Path(s.video_clip_path).resolve())]
                slide_is_clip.append(True)
            else:
                # Still image — loop to create infinite video stream.
                cmd += ["-loop", "1", "-i", str(img_paths[i])]
                slide_is_clip.append(False)

        voice_index = len(slides)
        cmd += ["-i", str(voice_path)]

        bgm_index: int | None = None
        if bgm_path or bgm_generate:
            bgm_index = voice_index + 1
            if bgm_path:
                cmd += ["-stream_loop", "-1", "-i", str(Path(bgm_path).resolve())]
            else:
                cmd += [
                    "-f",
                    "lavfi",
                    "-i",
                    "aevalsrc=0.25*sin(2*PI*110*t)+0.22*sin(2*PI*138.59*t)+0.20*sin(2*PI*164.81*t)+0.18*sin(2*PI*220*t):s=44100",
                ]

        # Video filter chain.
        v_filters: list[str] = []
        v_labels: list[str] = []
        fade_s = min(trans_s, 1.0)

        for i, dur in enumerate(durations):
            in_label = f"{i}:v"
            base_label = f"v{i}"
            scale_pad = (
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height}:(in_w-out_w)/2:(in_h-out_h)/2,"
                f"format=yuv420p,setsar=1,fps={fps}"
            )

            dur_s = float(dur)

            # ── Video clip slide: trim from clip offset, scale, no Ken Burns ──
            if slide_is_clip[i]:
                clip_start = float(slides[i].video_clip_start or 0.0)
                # Use video_clip_end to limit trim duration (clip may be shorter than slide).
                clip_end = float(slides[i].video_clip_end or dur_s)
                clip_dur = min(dur_s, clip_end)
                chain = (
                    f"[{in_label}]"
                    f"trim=start={clip_start:.3f}:duration={clip_dur:.3f},setpts=PTS-STARTPTS,"
                    # tpad pads with last frame if clip is slightly shorter than expected.
                    f"tpad=stop_mode=clone:stop_duration={max(0.0, dur_s - clip_dur + 0.1):.3f},"
                    f"trim=duration={dur_s:.3f},setpts=PTS-STARTPTS,"
                    f"{scale_pad}"
                )
                if trans == "fade" and fade_s > 0.0 and dur_s > (2 * fade_s + 0.05):
                    chain += f",fade=t=in:st=0:d={fade_s:.3f},fade=t=out:st={dur_s - fade_s:.3f}:d={fade_s:.3f}"
                chain += f"[{base_label}]"
                v_filters.append(chain)
                v_labels.append(f"[{base_label}]")
                continue

            # ── Still image slide (original logic) ──
            motion = str(getattr(slides[i], "motion", "") or "").strip().lower()
            if ken_burns and not motion:
                motion = "zoom_in"

            if ken_burns and motion and motion not in {"hold", "none", "static"}:
                frames = max(1, int(round(dur_s * float(fps))))
                denom = max(1, frames - 1)
                zoom_max = 1.08
                if motion in {"snap", "snap_zoom", "snapzoom"}:
                    zoom_max = 1.12
                zoom_delta = zoom_max - 1.0

                # Stable Ken Burns without zoompan (avoids common jitter/wobble artifacts).
                # We scale up deterministically per frame, force even dimensions, then crop back to output.
                prog = f"if(gte(n,{denom}),1,n/{denom})"

                if motion in {"snap", "snap_zoom", "snapzoom"}:
                    snap_d = f"max(1,{int(max(1.0, round(0.18 * denom)))})"
                    prog_eff = f"if(lte(n,{snap_d}),n/{snap_d},1)"
                else:
                    prog_eff = prog

                if motion in {"zoom_out", "out"}:
                    scale_expr = f"(1+{zoom_delta:.6f}*(1-{prog_eff}))"
                else:
                    scale_expr = f"(1+{zoom_delta:.6f}*{prog_eff})"

                # Pans keep a slightly larger scale and move the crop window.
                if motion in {"pan_lr", "pan", "pan_left_right", "left_right"}:
                    scale_expr = "1.10"
                    crop_x = f"2*trunc(((in_w-out_w)*{prog})/4)"
                    crop_y = "2*trunc((in_h-out_h)/4)"
                elif motion in {"pan_rl", "pan_right_left", "right_left"}:
                    scale_expr = "1.10"
                    crop_x = f"2*trunc(((in_w-out_w)*(1-{prog}))/4)"
                    crop_y = "2*trunc((in_h-out_h)/4)"
                else:
                    crop_x = "2*trunc((in_w-out_w)/4)"
                    crop_y = "2*trunc((in_h-out_h)/4)"

                scale_w = f"2*trunc(iw*{scale_expr}/2)"
                scale_h = f"2*trunc(ih*{scale_expr}/2)"
                effect = (
                    f"format=rgba,"
                    f"scale=w='{scale_w}':h='{scale_h}':eval=frame:flags=lanczos+accurate_rnd,"
                    f"crop={width}:{height}:x='{crop_x}':y='{crop_y}',"
                    f"format=yuv420p,setsar=1"
                )
                chain = f"[{in_label}]{scale_pad},{effect},trim=duration={dur_s:.3f},setpts=PTS-STARTPTS"
            else:
                chain = f"[{in_label}]{scale_pad},trim=duration={dur_s:.3f},setpts=PTS-STARTPTS"

            if trans == "fade" and fade_s > 0.0 and dur_s > (2 * fade_s + 0.05):
                chain += f",fade=t=in:st=0:d={fade_s:.3f},fade=t=out:st={dur_s - fade_s:.3f}:d={fade_s:.3f}"

            chain += f"[{base_label}]"
            v_filters.append(chain)
            v_labels.append(f"[{base_label}]")

        v_filters.append("".join(v_labels) + f"concat=n={len(v_labels)}:v=1:a=0[vout]")

        # ── Compute clip timeline ranges (for ducking narrator during clips) ──
        clip_ranges: list[tuple[float, float, int, bool]] = []  # (tl_start, tl_dur, input_idx, muted)
        cumulative_t = 0.0
        for i, dur in enumerate(durations):
            dur_f = float(dur)
            if slide_is_clip[i]:
                clip_ranges.append((cumulative_t, dur_f, i, slides[i].video_clip_mute))
            cumulative_t += dur_f

        # Audio filter chain.
        voice_label = f"{voice_index}:a"
        bgm_label = f"{bgm_index}:a" if bgm_index is not None else None
        pre_a_filters: list[str] = []  # extra audio filters added before _audio_filters
        clip_audio_labels: list[str] = []

        if clip_ranges:
            # 1) Mute narrator voice during clip playback ranges.
            #    volume=0 is enabled only while any clip is playing; otherwise full volume.
            between_parts = "+".join(
                f"between(t,{ts:.3f},{ts + td:.3f})" for ts, td, _, _ in clip_ranges
            )
            pre_a_filters.append(
                f"[{voice_label}]volume=0:enable='{between_parts}'[vduck]"
            )
            voice_label = "vduck"

            # 2) For non-muted clips, extract their audio and position on the timeline.
            for ci, (tl_start, tl_dur, inp_idx, muted) in enumerate(clip_ranges):
                if muted:
                    continue  # clip was trimmed with -an, no audio stream
                cs = float(slides[inp_idx].video_clip_start or 0.0)
                ce = float(slides[inp_idx].video_clip_end or tl_dur)
                cd = min(tl_dur, ce)
                delay_ms = int(round(tl_start * 1000.0))
                lbl = f"ca{ci}"
                pre_a_filters.append(
                    f"[{inp_idx}:a]"
                    f"atrim=start={cs:.3f}:duration={cd:.3f},asetpts=PTS-STARTPTS,"
                    f"aformat=sample_fmts=fltp:sample_rates=44100,"
                    f"adelay={delay_ms}|{delay_ms},"
                    f"apad=whole_dur={total_duration:.3f},"
                    f"atrim=duration={total_duration:.3f}"
                    f"[{lbl}]"
                )
                clip_audio_labels.append(f"[{lbl}]")

        a_frag, a_map = _audio_filters(use_duck=use_duck, voice_label=voice_label, bgm_label=bgm_label)

        # Assemble all filter fragments.
        filter_parts: list[str] = list(v_filters)
        if pre_a_filters:
            filter_parts.extend(pre_a_filters)
        if a_frag:
            filter_parts.append(a_frag)

        has_filter_audio = bool(a_frag)

        # 3) Mix clip audio into the final audio output.
        if clip_audio_labels:
            main_lbl = a_map if has_filter_audio else voice_label
            if not has_filter_audio:
                # Voice hasn't been formatted yet — prepare it for amix.
                filter_parts.append(
                    f"[{main_lbl}]aformat=sample_fmts=fltp:sample_rates=44100,"
                    f"apad,atrim=duration={total_duration:.3f}[vmain]"
                )
                main_lbl = "vmain"
            n = 1 + len(clip_audio_labels)
            filter_parts.append(
                f"[{main_lbl}]{''.join(clip_audio_labels)}"
                f"amix=inputs={n}:duration=first:dropout_transition=2:normalize=0,"
                f"alimiter=limit=0.97[afinal]"
            )
            a_map = "afinal"
            has_filter_audio = True
        elif clip_ranges and not has_filter_audio:
            # Voice was ducked but no clip audio to mix — still need filter output.
            filter_parts.append(
                f"[{voice_label}]aformat=sample_fmts=fltp:sample_rates=44100,"
                f"apad,atrim=duration={total_duration:.3f},alimiter=limit=0.97[afinal]"
            )
            a_map = "afinal"
            has_filter_audio = True

        cmd += [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vout]",
        ]

        if has_filter_audio:
            cmd += ["-map", f"[{a_map}]"]
        else:
            cmd += ["-map", f"{voice_index}:a:0"]

        cmd += [
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "cfr",
            "-r",
            str(int(fps)),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
        ]

        return cmd

    # If transitions, Ken Burns, or video clips are present, use the filter_complex path.
    has_video_clips = any(
        s.video_clip_path and Path(s.video_clip_path).exists() for s in slides
    )
    if trans is not None or ken_burns or has_video_clips:
        base = _build_with_filters(use_duck=bgm_duck)

    # Prefer H.264 for YouTube; fall back if encoder isn't available.
    # When video clips are present, don't use -tune stillimage.
    if has_video_clips:
        attempts = [
            ("libx264", ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]),
            ("mpeg4", ["-c:v", "mpeg4", "-q:v", "4"]),
        ]
    else:
        attempts = [
            ("libx264", ["-c:v", "libx264", "-tune", "stillimage"]),
            ("mpeg4", ["-c:v", "mpeg4", "-q:v", "4"]),
        ]

    last_stderr = ""
    for name, extra in attempts:
        cur_base = base
        for _ in range(2):
            cmd = cur_base + extra + [str(out_mp4)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                return

            last_stderr = proc.stderr or ""

            # If ducking was requested but ffmpeg lacks sidechaincompress, retry without ducking.
            if (bgm_path or bgm_generate) and bgm_duck and "sidechaincompress" in last_stderr and (
                "No such filter" in last_stderr or "not found" in last_stderr
            ):
                bgm_duck = False
                cur_base = _build_base(use_duck=False)
                continue

            # If it's not an encoder issue, don't retry.
            if not _is_encoder_error(last_stderr):
                raise RuntimeError(f"ffmpeg failed ({proc.returncode}):\n{last_stderr}")
            break

    raise RuntimeError(f"ffmpeg failed: could not find a working video encoder.\n{last_stderr}")
