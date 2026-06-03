import json
import logging
from datetime import datetime, timedelta

import paramiko
import sshtunnel

from lac_pipeline.events import build_event_tables
from lac_pipeline.geospatial import (
    add_ground_elevation,
    add_population_density,
    add_projected_coordinates,
)
from lac_pipeline.metrics import (
    collect_pipeline_metrics,
    empty_pipeline_metrics,
    write_metrics_json,
)
from lac_pipeline.obstacles import ClearanceConfig, add_obstacle_clearance, load_obstacles
from lac_pipeline.opensky import (
    GeographicBounds,
    build_operational_status_query,
    build_position_query,
    build_query_window,
    build_state_vectors_query,
    fetch_opensky_dataframe,
)
from lac_pipeline.publishing import (
    build_publish_target,
    ensure_publishable_manifest_state,
    mysql_engine_via_tunnel,
    publish_dataframes,
    publish_empty_day,
    reload_pythonanywhere,
    require_publish_allowed,
)
from lac_pipeline.runtime import (
    GracefulPipelineError,
    LockError,
    PipelineLock,
    configure_logging,
    install_max_runtime_guard,
    parse_runtime_settings,
    stage,
)
from lac_pipeline.trajectory import add_trajectory_columns, haversine
from lac_pipeline.transforms import merge_asof_by_icao
from lac_pipeline.validation import validate_pipeline_outputs


def main(argv: list[str] | None = None) -> int:
    run_lock = None
    logger = logging.getLogger("lac_pipeline.nightly")
    try:
        update_start_time = datetime.now()
        settings = parse_runtime_settings(argv)
        configure_logging(settings)
        logger = logging.getLogger("lac_pipeline.nightly")

        try:
            require_publish_allowed(
                settings.publish, settings.target, settings.confirm_production
            )
            run_lock = PipelineLock(
                settings.lock_file,
                settings.run_id,
                settings.stale_lock_seconds,
                logger,
            )
            run_lock.acquire()
            install_max_runtime_guard(settings.max_runtime_seconds, logger)
        except LockError as exc:
            logger.error("lock_unavailable error=%s", exc)
            return 75
        except Exception as exc:
            logger.error("startup_failed error=%s", exc)
            return 64

        LAT_MIN, LAT_MAX = 50.88385859501204322, 50.98427935836787128
        LON_MIN, LON_MAX = 6.85029965503896943, 7.005  # 7.03641128126701965
        ALT_MIN, ALT_MAX = (
            0,
            750,
        )
        # Updated from 700 m to 750 m for the CTR limit at 2500 ft plus margin.
        # The margin also accounts for the Cologne geoid height correction.

        TIME_BETWEEN_TRAJS = 30

        # SERA.5005(f)(1) criteria
        CONGESTED_ALERT_DISTANCE_M = 600  # alert distance wrt obstacles (should be 600)
        CONGESTED_ALERT_DELTA_HEIGHT_M = 300  # delta height (should be 300)

        # SERA.5005(f)(2) criteria
        NONCONGESTED_ALERT_DISTANCE_M = 150
        NONCONGESTED_ALERT_DELTA_HEIGHT_M = 150

        GEOID_HEIGHT_M = 47  # geoid height for Cologne

        sshtunnel.SSH_TIMEOUT = settings.ssh_timeout_seconds
        sshtunnel.TUNNEL_TIMEOUT = settings.tunnel_timeout_seconds

        MYSQL_secrets_json = "./mysql_secrets.json"
        PYA_secrets_json = "./PYA_secrets.json"

        MYSQL_creds = {}
        PYA_creds = {}
        ed25519_key = None

        if settings.publish:
            with open(MYSQL_secrets_json) as MYSQL_secrets:
                MYSQL_creds = json.load(MYSQL_secrets)

            with open(PYA_secrets_json) as PYA_secrets:
                PYA_creds = json.load(PYA_secrets)

            ed25519_key = paramiko.Ed25519Key(filename="./.ssh/id_ed25519")

        publish_target = build_publish_target(MYSQL_creds, settings.target)
        logger.info(
            "pipeline_start mode=%s target=%s publish=%s",
            "dry-run" if settings.dry_run else "publish",
            publish_target.name,
            settings.publish,
        )

        # Obtain and format the date to retrieve data for (3 days ago by default)
        if settings.target_date:
            two_days_ago = datetime.strptime(settings.target_date, "%Y-%m-%d")
        else:
            two_days_ago = datetime.now() - timedelta(days=3)
        date_string = two_days_ago.strftime("%Y-%m-%d")
        logger.info("target_date date=%s", date_string)
        query_window = build_query_window(two_days_ago)
        geographic_bounds = GeographicBounds(
            lat_min=LAT_MIN,
            lat_max=LAT_MAX,
            lon_min=LON_MIN,
            lon_max=LON_MAX,
        )

        if settings.publish:
            with stage(logger, "early_manifest_check"):
                with mysql_engine_via_tunnel(MYSQL_creds, ed25519_key, publish_target) as engine:
                    with engine.connect() as connection:
                        state = ensure_publishable_manifest_state(
                            connection, two_days_ago.date(), publish_target.table_names
                        )
                        if state == "complete":
                            logger.info(
                                "date_already_published date=%s target=%s",
                                date_string,
                                publish_target.name,
                            )
                            return 0

        # First query for State Vectors
        svdata4_query = build_state_vectors_query(
            query_window,
            geographic_bounds,
            ALT_MIN,
            ALT_MAX,
        )

        with stage(logger, "query_state_vectors"):
            svdata4_df = fetch_opensky_dataframe(
                "state_vectors_data4",
                svdata4_query,
                settings.query_attempts,
                settings.query_retry_delay_seconds,
                logger,
                cached=False,
                compress=True,
            )
        logger.info("query_rows table=state_vectors_data4 rows=%s", len(svdata4_df))

        if svdata4_df.empty:
            logger.warning("empty_source_day date=%s", date_string)
            if settings.validation_metrics_json:
                write_metrics_json(
                    settings.validation_metrics_json,
                    two_days_ago.date(),
                    empty_pipeline_metrics(),
                )
            if settings.publish:
                with stage(logger, "publish_empty_day"):
                    with mysql_engine_via_tunnel(
                        MYSQL_creds, ed25519_key, publish_target
                    ) as engine:
                        publish_empty_day(
                            engine,
                            publish_target,
                            two_days_ago.date(),
                            update_start_time,
                            logger,
                        )
            else:
                logger.info("dry_run_empty_day no_database_changes date=%s", date_string)
            logger.info("pipeline_complete status=empty_source_day")
            return 0

        # Save svdata4 pickle
        svdata4_df.to_pickle(f"./OSN_pickles/svdata4df_new_{date_string}.pkl")

        # Second Query for Ops Status
        icao_list = svdata4_df.icao24.unique()
        ops_sts_query = build_operational_status_query(icao_list, query_window)

        with stage(logger, "query_operational_status"):
            ops_sts_df = fetch_opensky_dataframe(
                "operational_status_data4",
                ops_sts_query,
                settings.query_attempts,
                settings.query_retry_delay_seconds,
                logger,
                cached=False,
            )
        logger.info("query_rows table=operational_status_data4 rows=%s", len(ops_sts_df))

        ops_sts_df["time"] = ops_sts_df["mintime"].astype("int64")

        # Save ops_sts pickle
        ops_sts_df.to_pickle(f"./OSN_pickles/opsstsdf_new_{date_string}.pkl")

        merged_df = merge_asof_by_icao(svdata4_df, ops_sts_df)

        # Third Query for Position data (to get the NIC)
        posdata4_query = build_position_query(
            icao_list,
            query_window,
            geographic_bounds,
        )

        with stage(logger, "query_position_data"):
            posdata4_df = fetch_opensky_dataframe(
                "position_data4",
                posdata4_query,
                settings.query_attempts,
                settings.query_retry_delay_seconds,
                logger,
                cached=False,
            )
        logger.info("query_rows table=position_data4 rows=%s", len(posdata4_df))

        posdata4_df["time"] = posdata4_df["mintime"].astype("int64")

        # Save ops_sts pickle
        posdata4_df.to_pickle(f"./OSN_pickles/posdata4df_new_{date_string}.pkl")

        final_df = merge_asof_by_icao(merged_df, posdata4_df)

        final_df = final_df.drop(columns=["hour", "mintime_x", "maxtime", "mintime_y"])

        ## Add DEM ground elevation information

        final_df = add_projected_coordinates(final_df)
        final_df = add_ground_elevation(
            final_df, "./resources/Cologne_DEM_merged_from_LAS_25x25.tif"
        )

        missing_elev_count = final_df["gnd_elev"].isnull().sum()
        logger.info("null_count column=gnd_elev rows=%s", missing_elev_count)
        final_df = final_df.dropna(subset=["gnd_elev"])

        ## Add population density

        final_df = add_population_density(
            final_df, "./resources/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_Cologne.tif"
        )

        ## Process trajectory distance information

        final_df = add_trajectory_columns(final_df, TIME_BETWEEN_TRAJS, haversine)

        ## Load obstacle information and check min height

        obs_df = load_obstacles("./resources/LAC_obstacles_v1.csv")
        clearance_config = ClearanceConfig(
            congested_alert_distance_m=CONGESTED_ALERT_DISTANCE_M,
            congested_alert_delta_height_m=CONGESTED_ALERT_DELTA_HEIGHT_M,
            noncongested_alert_distance_m=NONCONGESTED_ALERT_DISTANCE_M,
            noncongested_alert_delta_height_m=NONCONGESTED_ALERT_DELTA_HEIGHT_M,
            geoid_height_m=GEOID_HEIGHT_M,
        )
        final_df = add_obstacle_clearance(final_df, obs_df, clearance_config)

        final_df = final_df.drop(columns=["serials", "nacv"])

        missing_callsign = final_df["callsign"].isnull().sum()
        logger.info("null_count column=callsign rows=%s", missing_callsign)
        final_df = final_df.dropna(subset=["callsign"])

        ## Create candidate event tables

        inf_result, gnd_inf_result = build_event_tables(final_df, obs_df, haversine)

        ###

        gdf_record_count = len(final_df)
        inf_record_count = len(inf_result)
        gndinf_record_count = len(gnd_inf_result)
        processed_date = two_days_ago.date()

        ### Validate and optionally publish data

        logger.info(
            "output_row_counts main=%s inf=%s gndinf=%s",
            gdf_record_count,
            inf_record_count,
            gndinf_record_count,
        )

        with stage(logger, "validate_outputs"):
            validation_results = validate_pipeline_outputs(
                final_df, inf_result, gnd_inf_result, processed_date
            )
            for result in validation_results:
                logger.info(
                    "validation_result frame=%s rows=%s null_counts=%s",
                    result.name,
                    result.row_count,
                    result.null_counts,
                )

        if settings.validation_metrics_json:
            write_metrics_json(
                settings.validation_metrics_json,
                processed_date,
                collect_pipeline_metrics(final_df, inf_result, gnd_inf_result),
            )

        if settings.dry_run:
            if settings.show_results is not None:
                logger.info(
                    "dry_run_results\n%s",
                    format_dry_run_results(
                        inf_result,
                        gnd_inf_result,
                        settings.show_results,
                    ),
                )
            logger.info(
                "dry_run_complete no_database_changes target=%s date=%s",
                publish_target.name,
                processed_date,
            )
            logger.info("pipeline_complete status=dry_run")
            return 0

        with stage(logger, "publish_dataframes"):
            with mysql_engine_via_tunnel(MYSQL_creds, ed25519_key, publish_target) as engine:
                publish_counts = publish_dataframes(
                    engine,
                    publish_target,
                    final_df,
                    inf_result,
                    gnd_inf_result,
                    processed_date,
                    update_start_time,
                    logger,
                )
            logger.info(
                "publish_complete target=%s main=%s inf=%s gndinf=%s",
                publish_target.name,
                publish_counts.main_rows,
                publish_counts.inf_rows,
                publish_counts.gndinf_rows,
            )

        if settings.skip_reload:
            logger.info("pythonanywhere_reload_skipped")
        else:
            with stage(logger, "pythonanywhere_reload"):
                reload_pythonanywhere(PYA_creds, settings.http_timeout_seconds, logger)

        logger.info("pipeline_complete status=success")
        return 0
    except GracefulPipelineError as exc:
        logger.error(
            "pipeline_aborted status=%s exit_code=%s error=%s",
            exc.status,
            exc.exit_code,
            exc,
        )
        return exc.exit_code
    finally:
        if run_lock is not None:
            run_lock.release()


def format_dry_run_results(inf_df, gndinf_df, rows: int) -> str:
    row_limit = max(0, rows)
    total = len(inf_df) + len(gndinf_df)
    lines = [
        "detected_infractions "
        f"total={total} inf={len(inf_df)} gndinf={len(gndinf_df)}"
    ]
    if row_limit > 0:
        lines.extend(
            [
                f"inf first {row_limit} rows:",
                _format_table_head(inf_df, row_limit),
                f"gndinf first {row_limit} rows:",
                _format_table_head(gndinf_df, row_limit),
            ]
        )
    return "\n".join(lines)


def _format_table_head(frame, rows: int) -> str:
    if frame.empty:
        return "<empty>"
    return frame.head(rows).to_string(index=False)
