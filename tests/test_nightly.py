import pandas as pd

from lac_pipeline.nightly import format_dry_run_results


def test_format_dry_run_results_counts_and_limits_rows():
    inf_df = pd.DataFrame(
        {
            "inf_ref": ["inf-1", "inf-2"],
            "icao24": ["abc123", "def456"],
        }
    )
    gndinf_df = pd.DataFrame({"inf_ref": ["gnd-1"], "icao24": ["fed321"]})

    preview = format_dry_run_results(inf_df, gndinf_df, 1)

    assert "detected_infractions total=3 inf=2 gndinf=1" in preview
    assert "inf first 1 rows:" in preview
    assert "gndinf first 1 rows:" in preview
    assert "inf-1" in preview
    assert "inf-2" not in preview
    assert "gnd-1" in preview


def test_format_dry_run_results_handles_empty_tables():
    preview = format_dry_run_results(pd.DataFrame(), pd.DataFrame(), 2)

    assert "detected_infractions total=0 inf=0 gndinf=0" in preview
    assert preview.count("<empty>") == 2
