# Audio → Transcript → Licensed Images → Video (Python)

This project takes an input audio file (e.g. MP3), transcribes it with timestamps, downloads **real** images from **Wikimedia Commons** (license-aware), and renders a slideshow MP4 with the audio synced to the chosen transcript segments.

## What it does

- Input: `audio.mp3`
- Output: `output/video.mp4` + `output/timeline.json` + `output/attribution.txt`

## Setup

This project is easiest to run with `uv`.

1) Install uv (once):

```bash
pip install uv
```

2) Create the virtual environment + install deps:

```bash
uv sync --extra dev
```

Notes:
- Local transcription uses `openai-whisper`, which requires `ffmpeg` available on your machine.
- Video rendering uses an ffmpeg binary via `imageio-ffmpeg` (downloaded automatically).
- Transcripts are cached by audio filename + file stats in `.cache/transcripts/` to speed up reruns across different output folders.

## Run

```bash
uv run python -m videogenerator \
  --audio "path/to/audio.mp3" \
  --out "output" \
  --topic "Severance TV series" \
  --max-images 12
```

### Branding (intro/outro slates)

By default, the tool generates a simple intro + outro slate with a BR logo and a title derived from the transcript.

```bash
uv run python -m videogenerator \
  --audio severancereview_1min.mp3 \
  --out output_branded \
  --topic "Severance TV series" \
  --intro-seconds 2.5 \
  --outro-seconds 3.0 \
  --logo-scheme teal
```

All logo variants are saved into `output/assets/` as `brand_logo_br_<scheme>.png`.

### Add light background music (optional)

Mix a background track quietly behind your narration:

```bash
uv run python -m videogenerator \
  --audio severancereview_1min.mp3 \
  --out output_bgm \
  --topic "Severance (TV series)" \
  --max-images 6 \
  --bgm "path/to/background.mp3" \
  --bgm-volume 0.08
```

Or generate a simple ambient bed (no external file):

```bash
uv run python -m videogenerator \
  --audio severancereview_1min.mp3 \
  --out output_bgm_generated \
  --topic "Severance (TV series)" \
  --max-images 6 \
  --bgm-generate \
  --bgm-volume 0.06
```

Or use a built-in preset (copyright-free, generated locally):

```bash
uv run python -m videogenerator \
  --audio severancereview_1min.mp3 \
  --out output_bgm_elevator \
  --topic "Severance (TV series)" \
  --max-images 6 \
  --bgm-preset elevator \
  --bgm-volume 0.10
```

### Verify the output MP4

Local verification (no network): checks the MP4 has an audio stream, measures a quick audio peak, and extracts a few frames to see if they change.

```bash
uv run python -m videogenerator \
  --audio severancereview_1min.mp3 \
  --out output_verify \
  --topic "Severance (TV series)" \
  --image-provider serpapi \
  --max-images 6 \
  --verify-video
```

### Transcribe only

Write the transcript to `transcript.json` and `transcript.txt` in the output folder:

```bash
uv run python -m videogenerator \
  --audio severancereview_1min.mp3 \
  --out output_transcript_1min \
  --transcribe-only
```

You can also use the installed script:

```bash
uv run videogenerator --audio "path/to/audio.mp3" --out "output" --topic "Severance TV series" --max-images 12
```

## Important licensing note

This tool only downloads images with explicit license metadata (e.g. CC BY, CC BY-SA, Public Domain) from Wikimedia Commons and writes attribution info to `output/attribution.txt`. You are responsible for complying with each license (attribution, share-alike, etc.).

When using `--image-provider google_images`, the tool downloads images directly from the web via SerpAPI Google Images results. You are responsible for ensuring you have the rights to use those images and for complying with any licensing/attribution requirements.

This project does **not** include a mode that ignores licensing/copyright.

## SerpAPI (optional)

If you want Google Images discovery via SerpAPI, set `SERPAPI_API_KEY` (recommended: in a local `.env` file) and run:

Create `.env` (do not commit it):

```bash
echo SERPAPI_API_KEY=your_key_here > .env
```

```bash
uv run videogenerator --image-provider serpapi --audio "path/to/audio.mp3" --out output --topic "Severance (TV series)" --max-images 12
```

For safety, the SerpAPI provider is restricted to Wikimedia Commons hosts and still uses Commons license metadata.

### SerpAPI Google Images (license-filtered, not Commons-only)

If you want to use Google Images results directly (not restricted to Commons), use `--image-provider google_images`.
This mode does not perform license validation.

```bash
uv run videogenerator \
  --image-provider google_images \
  --audio "path/to/audio.mp3" \
  --out output_google \
  --topic "Severance (TV series)" \
  --max-images 12
```

## Structure

- `videogenerator/` core pipeline
- `tests/` small unit tests for segment selection
