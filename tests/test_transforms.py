import pandas as pd
from pandas.testing import assert_frame_equal

from lac_pipeline.transforms import merge_asof_by_icao


def test_merge_asof_by_icao_matches_legacy_loop():
    left = pd.DataFrame(
        {
            "icao24": ["bbb", "aaa", "bbb", "aaa"],
            "time": [10, 10, 30, 30],
            "lat": [1.0, 2.0, 3.0, 4.0],
        }
    )
    right = pd.DataFrame(
        {
            "icao24": ["aaa", "bbb", "aaa", "bbb"],
            "time": [5, 5, 25, 25],
            "quality": [1, 2, 3, 4],
        }
    )

    merged = merge_asof_by_icao(left, right)
    expected = _legacy_merge(left, right)

    assert_frame_equal(merged, expected)


def test_merge_asof_by_icao_keeps_left_rows_when_right_has_no_match():
    left = pd.DataFrame({"icao24": ["aaa"], "time": [10], "lat": [1.0]})
    right = pd.DataFrame({"icao24": ["bbb"], "time": [5], "quality": [2]})

    merged = merge_asof_by_icao(left, right)

    assert len(merged) == 1
    assert pd.isna(merged["quality"].iloc[0])


def test_merge_asof_by_icao_handles_empty_right_frame_with_schema():
    left = pd.DataFrame({"icao24": ["aaa"], "time": [10], "lat": [1.0]})
    right = pd.DataFrame(
        {
            "icao24": pd.Series(dtype=object),
            "time": pd.Series(dtype="int64"),
            "quality": pd.Series(dtype="float64"),
        }
    )

    merged = merge_asof_by_icao(left, right)

    assert len(merged) == 1
    assert "quality" in merged.columns
    assert pd.isna(merged["quality"].iloc[0])


def _legacy_merge(left, right):
    merged = pd.DataFrame()
    unique_icao24s = pd.concat([left["icao24"], right["icao24"]]).unique()

    for icao24 in unique_icao24s:
        sub_df1 = left[left["icao24"] == icao24].sort_values("time")
        sub_df2 = right[right["icao24"] == icao24].sort_values("time")
        sub_df1["icao24"] = sub_df1["icao24"].astype("object")
        sub_df2["icao24"] = sub_df2["icao24"].astype("object")
        sub_df1["time"] = sub_df1["time"].astype("int64")
        sub_df2["time"] = sub_df2["time"].astype("int64")
        merged_sub_df = pd.merge_asof(
            sub_df1, sub_df2, on="time", by="icao24", direction="backward"
        )
        merged = pd.concat([merged, merged_sub_df], ignore_index=True)

    return merged.sort_values(["icao24", "time"]).reset_index(drop=True)

