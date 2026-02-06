from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv


def main() -> None:
    load_dotenv(override=False)

    p = argparse.ArgumentParser(description="Debug SerpAPI -> Commons candidate retrieval")
    p.add_argument("--query", required=True, help="Search query")
    p.add_argument("--min-width", type=int, default=900, help="Minimum width")
    p.add_argument("--max-results", type=int, default=10, help="Max SerpAPI results to consider")
    args = p.parse_args()

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise SystemExit("SERPAPI_API_KEY is not set")

    from .serpapi_provider import search_commons_candidates_via_serpapi

    candidates = search_commons_candidates_via_serpapi(
        args.query,
        api_key=api_key,
        min_width=args.min_width,
        max_results=args.max_results,
    )

    print(f"query={args.query!r}")
    print(f"candidates={len(candidates)}")

    for i, c in enumerate(candidates):
        print(
            f"\n[{i}] title={c.get('title')}\n"
            f"  page_url={c.get('page_url')}\n"
            f"  image_url={c.get('image_url')}\n"
            f"  original_url={c.get('original_url')}\n"
            f"  size={c.get('width')}x{c.get('height')}\n"
            f"  license={c.get('license_name')}\n"
            f"  attribution={c.get('attribution')}"
        )


if __name__ == "__main__":
    main()
