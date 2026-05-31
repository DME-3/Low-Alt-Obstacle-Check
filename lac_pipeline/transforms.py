from __future__ import annotations

import pandas as pd


def merge_asof_by_icao(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str = "time",
    by: str = "icao24",
    direction: str = "backward",
) -> pd.DataFrame:
    """Merge ADS-B frames by ICAO without repeated per-aircraft concat loops."""
    left_frame = left.copy()
    right_frame = right.copy()

    left_frame[by] = left_frame[by].astype("object")
    right_frame[by] = right_frame[by].astype("object")
    left_frame[on] = left_frame[on].astype("int64")
    right_frame[on] = right_frame[on].astype("int64")

    # pandas requires the asof key to be globally sorted even when `by` is used.
    left_frame = left_frame.sort_values([on, by])
    right_frame = right_frame.sort_values([on, by])

    merged = pd.merge_asof(
        left_frame,
        right_frame,
        on=on,
        by=by,
        direction=direction,
    )
    return merged.sort_values([by, on]).reset_index(drop=True)

