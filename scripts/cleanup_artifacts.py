#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_TARGETS = {
    "OSN_pickles": ["*.pkl"],
    "dataframes": ["*.json"],
    "data_baseline": ["*.json"],
    "reprocessing": ["*.json"],
}


def main() -> int:
    from lac_pipeline.retention import cleanup_candidates, find_retention_candidates

    parser = argparse.ArgumentParser(
        description="Dry-run by default cleanup for generated ADS-B pipeline artifacts."
    )
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--older-than-days", type=int, default=120)
    parser.add_argument("--execute", action="store_true", help="Actually delete files.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("cleanup_artifacts")

    repo_root = Path(args.root)
    all_candidates = []
    for relative_dir, patterns in DEFAULT_TARGETS.items():
        root = repo_root / relative_dir
        candidates = find_retention_candidates(root, patterns, args.older_than_days)
        all_candidates.extend(candidates)
        logger.info(
            "retention_scan directory=%s candidates=%s",
            root,
            len(candidates),
        )
        if args.verbose:
            for candidate in candidates:
                logger.info(
                    "candidate path=%s age_days=%.1f size_bytes=%s",
                    candidate.path,
                    candidate.age_days,
                    candidate.size_bytes,
                )

    candidate_bytes = sum(candidate.size_bytes for candidate in all_candidates)
    removed_count, removed_bytes = cleanup_candidates(all_candidates, args.execute)
    logger.info(
        "retention_summary mode=%s candidates=%s candidate_bytes=%s removed=%s removed_bytes=%s",
        "execute" if args.execute else "dry-run",
        len(all_candidates),
        candidate_bytes,
        removed_count,
        removed_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

