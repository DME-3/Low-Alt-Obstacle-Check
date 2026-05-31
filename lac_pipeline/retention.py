from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetentionCandidate:
    path: Path
    age_days: float
    size_bytes: int


def find_retention_candidates(
    root: Path,
    patterns: list[str],
    older_than_days: int,
    now: float | None = None,
) -> list[RetentionCandidate]:
    if older_than_days < 0:
        raise ValueError("older_than_days must be non-negative")

    now = time.time() if now is None else now
    cutoff_seconds = older_than_days * 24 * 60 * 60
    candidates: list[RetentionCandidate] = []

    if not root.exists():
        return candidates

    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            stat = path.stat()
            age_seconds = now - stat.st_mtime
            if age_seconds >= cutoff_seconds:
                candidates.append(
                    RetentionCandidate(
                        path=path,
                        age_days=age_seconds / (24 * 60 * 60),
                        size_bytes=stat.st_size,
                    )
                )

    return sorted(candidates, key=lambda candidate: str(candidate.path))


def cleanup_candidates(
    candidates: list[RetentionCandidate],
    execute: bool,
) -> tuple[int, int]:
    removed_count = 0
    removed_bytes = 0
    if not execute:
        return removed_count, removed_bytes

    for candidate in candidates:
        candidate.path.unlink()
        removed_count += 1
        removed_bytes += candidate.size_bytes
    return removed_count, removed_bytes

