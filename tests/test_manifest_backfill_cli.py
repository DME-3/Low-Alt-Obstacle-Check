from backfill_manifest import build_parser


def test_manifest_backfill_parser_defaults_to_test_target_and_dry_run():
    args = build_parser().parse_args(
        ["--start-date", "2026-05-29", "--end-date", "2026-05-29"]
    )

    assert args.target == "test"
    assert args.execute is False
    assert args.confirm_production is False


def test_manifest_backfill_parser_accepts_force_duplicate():
    args = build_parser().parse_args(
        [
            "--start-date",
            "2026-05-29",
            "--end-date",
            "2026-05-29",
            "--force-duplicate",
        ]
    )

    assert args.force_duplicate is True
