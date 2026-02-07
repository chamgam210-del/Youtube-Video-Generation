from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import os
import shutil
import time

from tqdm import tqdm

from .audio import get_audio_duration_seconds
from .segments import merge_short_segments, select_evenly_spaced
from .transcribe import transcribe_cached, write_transcript_files
from .models import Slide
from .utils import ensure_dir, extract_keywords, read_json, write_json
from .wikimedia import download_image, search_commons_image, search_commons_images


def run(
    *,
    audio_path: str | Path,
    out_dir: str | Path,
    topic: str | None = None,
    video_type: str = "review",  # review|explainer|shorts|auto
    image_provider: str = "wikimedia",
    serpapi_api_key: str | None = None,
    max_images: int = 12,
    min_seg_seconds: float = 6.0,
    whisper_model: str = "small",
    min_image_width: int = 900,
    video_width: int = 1920,
    video_height: int = 1080,
    cache_transcript: bool = True,
    cache_dir: str | Path | None = None,
    storyboard: str = "auto",  # auto|llm|heuristic
    llm_model: str = "gpt-4o-mini",
    llm_pick_images: bool = True,
    reuse_images: bool = True,
) -> Path:
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)

    audio_stat = audio_path.stat()

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

    vt = (video_type or "review").strip().lower()
    if vt not in {"review", "explainer", "shorts", "auto"}:
        vt = "review"

    # For non-review videos, we want a stable topic hint to keep image searches on the right subject.
    effective_topic: str | None = (topic or "").strip() or None
    topic_type: str | None = None
    if vt in {"explainer", "shorts", "auto"} and not effective_topic:
        if use_llm:
            try:
                from .llm_storyboard import infer_topic_with_llm

                inferred = infer_topic_with_llm(merged, model=llm_model)
                effective_topic = inferred.topic.strip() or None
                topic_type = inferred.topic_type
            except Exception:
                effective_topic = None
        if not effective_topic:
            from .llm_storyboard import infer_topic_fallback

            inferred = infer_topic_fallback(audio_stem=audio_path.stem)
            effective_topic = inferred.topic.strip() or None
            topic_type = inferred.topic_type

    # Heuristic: if transcript strongly suggests TV content, treat as tv_show.
    if vt in {"explainer", "shorts"} and (topic_type is None or topic_type == "other"):
        try:
            tail = " ".join(s.text for s in merged[-25:]).lower()
        except Exception:
            tail = ""
        tv_hints = (
            "tv",
            "tv show",
            "series",
            "episode",
            "season",
            "cast",
            "character",
            "apple tv",
            "netflix",
            "hbo",
        )
        if any(h in tail for h in tv_hints):
            topic_type = "tv_show"

    use_llm = storyboard in {"llm", "auto"}
    # planned items: (start, end, query, headline, subhead)
    planned: list[tuple[float, float, str, str, str | None]] = []

    if vt == "auto":
        if use_llm:
            try:
                from .llm_storyboard import classify_transcript_kind_with_llm

                vt = classify_transcript_kind_with_llm(merged, topic=topic, model=llm_model)
            except Exception:
                from .llm_storyboard import classify_transcript_kind_fallback

                vt = classify_transcript_kind_fallback(merged)
        else:
            from .llm_storyboard import classify_transcript_kind_fallback

            vt = classify_transcript_kind_fallback(merged)

    if use_llm:
        try:
            if vt in {"explainer", "shorts"}:
                from .llm_storyboard import plan_rich_slides_with_llm

                rich = plan_rich_slides_with_llm(
                    merged,
                    audio_duration=audio_duration,
                    topic=effective_topic,
                    max_slides=max_images,
                    kind=("shorts" if vt == "shorts" else "explainer"),
                    model=llm_model,
                )
                planned = [(s.start, s.end, s.query, s.headline, s.subhead) for s in rich]
            else:
                from .llm_storyboard import plan_slides_with_llm

                # Give the LLM a condensed transcript (merged segments) so it can pick good cut points.
                story = plan_slides_with_llm(
                    merged,
                    audio_duration=audio_duration,
                    topic=effective_topic,
                    max_images=max_images,
                    model=llm_model,
                )
                planned = [(s.start, s.end, s.query, "", None) for s in story]
        except Exception:
            if storyboard == "llm":
                raise
            planned = []

    if not planned and storyboard == "llm":
        raise RuntimeError("LLM storyboard produced no slides")

    if not planned:
        picked = select_evenly_spaced(merged, max_items=max_images, audio_duration=audio_duration)
        planned = [(s.start, s.end, "", "", None) for s in picked]

    slides: list[Slide] = []
    attribution_lines: list[str] = []

    def _find_reuse_source_dir(*, audio_stem: str) -> Path | None:
        if not reuse_images:
            return None

        stem = (audio_stem or "").strip().lower()
        if not stem:
            return None

        candidates: list[tuple[float, Path]] = []
        cwd = Path.cwd()
        for d in cwd.iterdir():
            if not d.is_dir():
                continue
            name = d.name.lower()
            if not name.startswith("output"):
                continue
            if d.resolve() == out_dir.resolve():
                continue
            if not (d / "assets").is_dir():
                continue
            if not (d / "timeline.json").exists():
                continue

            # Prefer exact match using run_meta.json when present.
            meta_ok = False
            meta_path = d / "run_meta.json"
            if meta_path.exists():
                try:
                    meta = read_json(meta_path)
                    if isinstance(meta, dict):
                        if (meta.get("audio_name") == audio_path.name) or (str(meta.get("audio_stem") or "").strip().lower() == stem):
                            # If size/mtime were recorded, check them too.
                            sz = meta.get("audio_size")
                            mt = meta.get("audio_mtime")
                            if sz is None or int(sz) == int(audio_stat.st_size):
                                if mt is None or float(mt) == float(audio_stat.st_mtime):
                                    meta_ok = True
                except Exception:
                    meta_ok = False

            # Back-compat: older outputs without run_meta.json fall back to folder-name matching.
            if not meta_ok:
                if stem not in name:
                    continue
            try:
                ts = (d / "timeline.json").stat().st_mtime
            except Exception:
                ts = 0.0
            candidates.append((ts, d))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    reuse_source_dir = _find_reuse_source_dir(audio_stem=audio_path.stem)
    reuse_image_paths: list[Path] = []

    if reuse_source_dir is not None:
        src_assets = reuse_source_dir / "assets"
        # Prefer slide assets (sXX_*.png) and seed assets; avoid logos/slates.
        preferred = []
        fallback = []
        for p in sorted(src_assets.glob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            nm = p.name.lower()
            if nm.startswith("s") or nm.startswith("seed"):
                preferred.append(p)
            else:
                fallback.append(p)

        pick = preferred if preferred else fallback
        for p in pick:
            dst = assets_dir / p.name
            if not dst.exists():
                try:
                    shutil.copy2(p, dst)
                except Exception:
                    continue
            reuse_image_paths.append(dst)

        # If we successfully reused images, carry forward attribution when available.
        if reuse_image_paths and (reuse_source_dir / "attribution.txt").exists():
            try:
                shutil.copy2(reuse_source_dir / "attribution.txt", out_dir / "attribution.txt")
            except Exception:
                pass

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
    if (reuse_source_dir is None) and topic:
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

    for i, (start, end, llm_query, headline, subhead) in enumerate(tqdm(planned, desc="Finding images")):
        # If LLM provided an explicit query, trust it. Otherwise derive from local keywords.
        q_main = (llm_query or "").strip()
        # Keep explainer/shorts queries anchored on the topic.
        if vt in {"explainer", "shorts"} and effective_topic:
            if effective_topic.lower() not in q_main.lower():
                q_main = f"{effective_topic} {q_main}".strip()
            # If this looks like TV content, bias toward stills/cast.
            if (topic_type == "tv_show") and not any(k in q_main.lower() for k in ("still", "stills", "cast", "scene")):
                q_main = f"{q_main} TV series still".strip()
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

        # Reuse existing downloaded assets when possible (no SerpAPI calls).
        if reuse_image_paths:
            p = reuse_image_paths[i % len(reuse_image_paths)]
            image_path = str(p)

        for q in ([] if image_path else queries):
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

        # For explainer/shorts, render a slide card with on-screen text.
        if vt in {"explainer", "shorts"}:
            ht = (headline or "").strip()
            sh = (subhead or "").strip() if subhead else None
            if ht:
                try:
                    from .slide_cards import SlideCardSpec, render_slide_card

                    card_path = assets_dir / f"card_{i:02d}.png"
                    rendered = render_slide_card(
                        background_image=image_path,
                        out_path=card_path,
                        spec=SlideCardSpec(headline=ht, subhead=sh),
                        width=int(video_width),
                        height=int(video_height),
                    )
                    slide = Slide(
                        start=slide.start,
                        end=slide.end,
                        image_path=str(rendered),
                        query=slide.query,
                        source_page=slide.source_page,
                        image_url=slide.image_url,
                        license_name=slide.license_name,
                        license_url=slide.license_url,
                        attribution=slide.attribution,
                    )
                except Exception:
                    pass
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

    # Attribution: if reused, we may have already copied attribution.txt above.
    if not (out_dir / "attribution.txt").exists():
        (out_dir / "attribution.txt").write_text("\n".join(attribution_lines) + "\n", encoding="utf-8")

    # Minimal run metadata to help future reuse/matching.
    try:
        write_json(
            out_dir / "run_meta.json",
            {
                "audio_name": audio_path.name,
                "audio_stem": audio_path.stem,
                "audio_size": int(audio_stat.st_size),
                "audio_mtime": float(audio_stat.st_mtime),
                "video_type": vt,
                "video_width": int(video_width),
                "video_height": int(video_height),
                "topic": effective_topic,
                "topic_type": topic_type,
                "image_provider": image_provider,
                "max_images": int(max_images),
                "min_image_width": int(min_image_width),
                "reused_from": str(reuse_source_dir) if reuse_source_dir else None,
                "created_at": time.time(),
                "cwd": os.getcwd(),
            },
        )
    except Exception:
        pass

    return out_dir
