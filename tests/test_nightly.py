import pandas as pd
from trino import exceptions as trino_exceptions

from lac_pipeline.nightly import format_dry_run_results, main
from lac_pipeline.opensky import OpenSkyUnavailableError


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


def test_main_exits_gracefully_when_opensky_unavailable(monkeypatch, tmp_path, capsys):
    def unavailable_fetch(*args, **kwargs):
        raise OpenSkyUnavailableError(
            "state_vectors_data4",
            2,
            trino_exceptions.HttpError("error 404: b'<title>404 Not Found</title>'"),
        )

    monkeypatch.setattr("lac_pipeline.nightly.fetch_opensky_dataframe", unavailable_fetch)
    lock_path = tmp_path / "nightly.lock"
    exit_code = main(
        [
            "--date",
            "2026-05-31",
            "--lock-file",
            str(lock_path),
            "--max-runtime-seconds",
            "0",
            "--query-retry-delay-seconds",
            "0",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 75
    assert not lock_path.exists()
    assert (
        "stage_failed name=query_state_vectors error=OpenSky/Trino source unavailable"
        in captured.err
    )
    assert "pipeline_aborted status=opensky_unavailable exit_code=75" in captured.err
    assert "Traceback" not in captured.err
