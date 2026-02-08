from __future__ import annotations

import math
import subprocess
import wave
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
            wf.writeframesraw(int.to_bytes(s_l & 0xFFFF, 2, "little", signed=False) + int.to_bytes(s_r & 0xFFFF, 2, "little", signed=False))

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
            wf.writeframesraw(
                int.to_bytes(s_l & 0xFFFF, 2, "little", signed=False)
                + int.to_bytes(s_r & 0xFFFF, 2, "little", signed=False)
            )

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
        elif preset in ("elevator", "elevator_music"):
            # Generate a short loop-friendly WAV and loop it in ffmpeg.
            # Important: keep this short (we already `-stream_loop -1` it),
            # otherwise generating a full-length WAV in Python is extremely slow.
            bgm_wav = out_mp4.parent / "bgm_elevator.wav"
            _generate_elevator_music_wav(out_wav=bgm_wav, seconds=32.0)
            bgm_path = bgm_wav
        elif preset in ("creepy", "horror", "spooky"):
            bgm_wav = out_mp4.parent / "bgm_creepy.wav"
            _generate_creepy_music_wav(out_wav=bgm_wav, seconds=32.0)
            bgm_path = bgm_wav
        else:
            raise ValueError(f"Unknown bgm_preset: {bgm_preset!r}")

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
                    f"[{bgm_label}]aformat=sample_fmts=fltp:sample_rates=44100,lowpass=f=8000,volume={bgm_vol:.4f},apad,atrim=duration={total_duration:.3f}[bgm];"
                    "[bgm][voice_sc]sidechaincompress=threshold=0.02:ratio=12:attack=5:release=400[bgmduck];"
                    "[voice_mix][bgmduck]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,alimiter=limit=0.97[aout]"
                )
            else:
                frag = (
                    f"[{voice_label}]aformat=sample_fmts=fltp:sample_rates=44100,{adelay}apad,atrim=duration={total_duration:.3f}[voice];"
                    f"[{bgm_label}]aformat=sample_fmts=fltp:sample_rates=44100,lowpass=f=8000,volume={bgm_vol:.4f},apad,atrim=duration={total_duration:.3f}[bgm];"
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
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
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

        for p in img_paths:
            # -loop 1 turns an image into an infinite video stream; we trim each slide in the filtergraph.
            cmd += ["-loop", "1", "-i", str(p)]

        voice_index = len(img_paths)
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
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1,fps={fps}"
            )

            dur_s = float(dur)
            if ken_burns:
                frames = max(1, int(round(dur_s * float(fps))))
                denom = max(1, frames - 1)
                zoom_max = 1.03
                zoom_delta = zoom_max - 1.0

                # Stable Ken Burns without zoompan (avoids common jitter/wobble artifacts).
                # We scale up deterministically per frame, force even dimensions, then crop back to output.
                prog = f"if(gte(n,{denom}),1,n/{denom})"
                scale_w = f"2*trunc(iw*(1+{zoom_delta:.6f}*{prog})/2)"
                scale_h = f"2*trunc(ih*(1+{zoom_delta:.6f}*{prog})/2)"
                effect = (
                    f"format=rgba,"
                    f"scale=w='{scale_w}':h='{scale_h}':eval=frame:flags=lanczos+accurate_rnd,"
                    f"crop={width}:{height}:x='2*trunc((in_w-out_w)/4)':y='2*trunc((in_h-out_h)/4)',"
                    f"format=yuv420p"
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

        # Audio filter chain.
        voice_label = f"{voice_index}:a"
        bgm_label = f"{bgm_index}:a" if bgm_index is not None else None
        a_frag, a_map = _audio_filters(use_duck=use_duck, voice_label=voice_label, bgm_label=bgm_label)

        filter_parts: list[str] = []
        filter_parts.extend(v_filters)
        if a_frag:
            filter_parts.append(a_frag)

        cmd += [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vout]",
        ]

        if a_frag:
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

    # Ken Burns was removed/disabled due to jitter on common players.
    ken_burns = False

    # If transitions are enabled, use the filter_complex path.
    if trans is not None:
        base = _build_with_filters(use_duck=bgm_duck)

    # Prefer H.264 for YouTube; fall back if encoder isn't available.
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
