from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib

from tqdm import tqdm

from .audio import get_audio_duration_seconds
from .segments import merge_short_segments, select_evenly_spaced
from .transcribe import transcribe_cached, write_transcript_files
from .models import Slide
from .utils import ensure_dir, extract_keywords, write_json
from .wikimedia import download_image, search_commons_image, search_commons_images


def run(
    *,
    audio_path: str | Path,
    out_dir: str | Path,
    topic: str | None = None,
    image_provider: str = "wikimedia",
    serpapi_api_key: str | None = None,
    max_images: int = 12,
    min_seg_seconds: float = 6.0,
    whisper_model: str = "small",
    min_image_width: int = 900,
    cache_transcript: bool = True,
    cache_dir: str | Path | None = None,
    storyboard: str = "auto",  # auto|llm|heuristic
    llm_model: str = "gpt-4o-mini",
    llm_pick_images: bool = True,
) -> Path:
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)

    assets_dir = ensure_dir(out_dir / "assets")

    audio_duration = get_audio_duration_seconds(audio_path)

    # Default transcript cache should be stable across output folders.
    effective_cache_dir = Path(cache_dir) if cache_dir else (Path.cwd() / ".cache" / "transcripts")
    segments = transcribe_cached(
        audio_path,
        model_name=whisper_model,
        cache_dir=effective_cache_dir,
        use_cache=cache_transcript,
    )

    # Always write transcript files into the output folder for convenience.
    try:
        write_transcript_files(segments, out_dir=out_dir)
    except Exception:
        pass

    # If whisper produced a 0-length fallback segment, expand to full duration
    if len(segments) == 1 and segments[0].start == 0.0 and segments[0].end == 0.0:
        segments = [segments[0].__class__(start=0.0, end=audio_duration, text=segments[0].text)]

    merged = merge_short_segments(segments, min_seconds=min_seg_seconds)

    use_llm = storyboard in {"llm", "auto"}
    planned: list[tuple[float, float, str]] = []

    if use_llm:
        from .llm_storyboard import plan_slides_with_llm

        try:
            # Give the LLM a condensed transcript (merged segments) so it can pick good cut points.
            story = plan_slides_with_llm(
                merged,
                audio_duration=audio_duration,
                topic=topic,
                max_images=max_images,
                model=llm_model,
            )
            planned = [(s.start, s.end, s.query) for s in story]
        except Exception:
            if storyboard == "llm":
                raise
            planned = []

    if not planned and storyboard == "llm":
        raise RuntimeError("LLM storyboard produced no slides")

    if not planned:
        picked = select_evenly_spaced(merged, max_items=max_images, audio_duration=audio_duration)
        planned = [(s.start, s.end, "") for s in picked]

    slides: list[Slide] = []
    attribution_lines: list[str] = []

    last_image_path: str | None = None
    last_info: dict | None = None
    used_source_pages: list[str] = []
    used_image_hashes: set[str] = set()

    def _hash_file(path_str: str) -> str:
        b = Path(path_str).read_bytes()
        return hashlib.sha256(b).hexdigest()

    def _search_candidates(qstr: str) -> list[dict]:
        if not qstr:
            return []
        if image_provider == "serpapi":
            if not serpapi_api_key:
                raise RuntimeError("--image-provider serpapi requires --serpapi-key or SERPAPI_API_KEY")
            from .serpapi_provider import search_commons_candidates_via_serpapi

            return search_commons_candidates_via_serpapi(
                qstr,
                api_key=serpapi_api_key,
                min_width=min_image_width,
                max_results=10,
            )
        if image_provider == "google_images":
            if not serpapi_api_key:
                raise RuntimeError("--image-provider google_images requires --serpapi-key or SERPAPI_API_KEY")
            from .serpapi_provider import search_google_images_candidates_via_serpapi

            return search_google_images_candidates_via_serpapi(
                qstr,
                api_key=serpapi_api_key,
                min_width=min_image_width,
                max_results=10,
            )
        return search_commons_images(qstr, min_width=min_image_width, max_results=10)

    def _window_text(start: float, end: float) -> str:
        text = ""
        for s in merged:
            if s.end <= start:
                continue
            if s.start >= end:
                break
            text += " " + s.text
        return text.strip()

    def _pick_info_with_llm(*, slide_query: str, window_text: str, candidates: list[dict]) -> dict | None:
        if not candidates:
            return None
        if not (use_llm and llm_pick_images):
            # Deterministic fallback: pick first candidate not already used.
            for c in candidates:
                if c.get("page_url") and c.get("page_url") in used_source_pages:
                    continue
                return c
            return candidates[0]

        from .llm_storyboard import pick_image_with_llm

        idx = pick_image_with_llm(
            slide_query=slide_query,
            window_text=window_text,
            candidates=candidates,
            used_source_pages=used_source_pages,
            model=llm_model,
        )
        return candidates[idx]

    def _download_unique(info: dict, *, prefix: str) -> str | None:
        dl_url = info.get("original_url") or info.get("image_url")
        if not dl_url:
            return None
        p = download_image(dl_url, assets_dir, prefix=prefix)
        h = _hash_file(str(p))
        if h in used_image_hashes:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
            return None
        used_image_hashes.add(h)
        return str(p)

    # Seed image: try to download at least one image for the topic.
    if topic:
        try:
            seed_candidates = _search_candidates(topic)
            seed = seed_candidates[0] if seed_candidates else None
            if seed and seed.get("image_url"):
                p = download_image(seed["image_url"], assets_dir, prefix="seed")
                last_image_path = str(p)
                last_info = seed
                if seed.get("page_url"):
                    used_source_pages.append(str(seed.get("page_url")))
                try:
                    used_image_hashes.add(_hash_file(last_image_path))
                except Exception:
                    pass
        except Exception:
            last_image_path = None
            last_info = None

    for i, (start, end, llm_query) in enumerate(tqdm(planned, desc="Finding images")):
        # If LLM provided an explicit query, trust it. Otherwise derive from local keywords.
        q_main = (llm_query or "").strip()
        wtext = _window_text(float(start), float(end))

        keywords: list[str] = []

        if storyboard == "llm" and not q_main:
            raise RuntimeError("LLM slide query is empty")

        if not q_main:
            # Find the transcript segment overlapping this window to derive keywords.
            keywords = extract_keywords(wtext, max_words=8)
            q_main = " ".join([topic] + keywords) if topic else " ".join(keywords)
            q_main = q_main.strip() or (topic or "")

        # Avoid overly-specific queries causing 0 results.
        if storyboard == "llm":
            queries = [q_main]
        else:
            queries: list[str] = []
            if q_main:
                queries.append(q_main)
            if topic:
                queries.append(topic)
                if keywords:
                    queries.append(" ".join([topic] + keywords[:3]))
            if keywords:
                queries.append(" ".join(keywords[:5]))
            # de-dup while preserving order
            seen_q: set[str] = set()
            queries = [x for x in queries if not (x in seen_q or seen_q.add(x))]

        info = None
        image_path = None
        winning_query = q_main

        for q in queries:
            try:
                candidates = _search_candidates(q)
            except Exception:
                candidates = []

            # If possible, avoid already-used pages.
            if candidates:
                candidates = [c for c in candidates if not (c.get("page_url") in used_source_pages)] + [
                    c for c in candidates if c.get("page_url") in used_source_pages
                ]

            picked = None
            try:
                picked = _pick_info_with_llm(slide_query=q, window_text=wtext, candidates=candidates)
            except Exception:
                picked = candidates[0] if candidates else None

            if picked:
                picked["_query"] = q
                winning_query = q
                # Try to download a unique image; if duplicate, try the next candidates.
                attempts = [picked] + [c for c in candidates if c is not picked]
                for cand in attempts[:10]:
                    try:
                        path_str = _download_unique(cand, prefix=f"s{i:02d}")
                    except Exception:
                        path_str = None
                    if path_str:
                        info = cand
                        image_path = path_str
                        if cand.get("page_url"):
                            used_source_pages.append(str(cand.get("page_url")))
                        last_info = cand
                        break

            if image_path:
                break

        if not image_path:
            # No image found; reuse last successful image (keeps timing valid)
            if last_image_path is None:
                continue
            image_path = last_image_path
            # Carry forward metadata so timeline/attribution stays accurate.
            info = last_info

        last_image_path = image_path

        slide = Slide(
            start=float(start),
            end=float(end),
            image_path=image_path,
            query=str(winning_query or q_main),
            source_page=(info or {}).get("page_url"),
            image_url=(info or {}).get("image_url"),
            license_name=(info or {}).get("license_name"),
            license_url=(info or {}).get("license_url"),
            attribution=(info or {}).get("attribution"),
        )
        slides.append(slide)

        if slide.source_page or slide.license_name:
            attribution_lines.append(
                f"{Path(slide.image_path).name} | {slide.license_name or ''} | {slide.license_url or ''} | {slide.source_page or ''} | {slide.attribution or ''}".strip()
            )

    if not slides:
        raise RuntimeError(
            "No images could be found/downloaded for the selected segments. Try a different --topic, increase --max-images, or lower --min-image-width."
        )

    # Ensure the slideshow covers the full audio duration contiguously.
    slides.sort(key=lambda s: s.start)
    stitched: list[Slide] = []
    cur = 0.0
    for idx, s in enumerate(slides):
        start = max(cur, float(s.start))
        # Provisional end; will be overwritten for all but last.
        end = max(start + 0.1, float(s.end))
        stitched.append(
            Slide(
                start=start,
                end=end,
                image_path=s.image_path,
                query=s.query,
                source_page=s.source_page,
                image_url=s.image_url,
                license_name=s.license_name,
                license_url=s.license_url,
                attribution=s.attribution,
            )
        )
        cur = end

    # Make ends align to next start, filling any gaps.
    for i in range(len(stitched) - 1):
        stitched[i] = Slide(
            start=stitched[i].start,
            end=max(stitched[i].start + 0.1, stitched[i + 1].start),
            image_path=stitched[i].image_path,
            query=stitched[i].query,
            source_page=stitched[i].source_page,
            image_url=stitched[i].image_url,
            license_name=stitched[i].license_name,
            license_url=stitched[i].license_url,
            attribution=stitched[i].attribution,
        )

    stitched[-1] = Slide(
        start=stitched[-1].start,
        end=float(audio_duration),
        image_path=stitched[-1].image_path,
        query=stitched[-1].query,
        source_page=stitched[-1].source_page,
        image_url=stitched[-1].image_url,
        license_name=stitched[-1].license_name,
        license_url=stitched[-1].license_url,
        attribution=stitched[-1].attribution,
    )

    slides = stitched

    if slides[0].start > 0:
        slides = [
            Slide(
                start=0.0,
                end=float(slides[0].start),
                image_path=slides[0].image_path,
                query=slides[0].query,
                source_page=slides[0].source_page,
                image_url=slides[0].image_url,
                license_name=slides[0].license_name,
                license_url=slides[0].license_url,
                attribution=slides[0].attribution,
            )
        ] + slides

    write_json(out_dir / "timeline.json", [asdict(s) for s in slides])
    (out_dir / "attribution.txt").write_text("\n".join(attribution_lines) + "\n", encoding="utf-8")

    return out_dir
