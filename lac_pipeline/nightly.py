import json
import logging
from datetime import datetime, timedelta

import paramiko
import sshtunnel
from pyopensky.trino import Trino

from lac_pipeline.events import build_event_tables
from lac_pipeline.geospatial import (
    add_ground_elevation,
    add_population_density,
    add_projected_coordinates,
)
from lac_pipeline.obstacles import ClearanceConfig, add_obstacle_clearance, load_obstacles
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
    LockError,
    PipelineLock,
    configure_logging,
    install_max_runtime_guard,
    parse_runtime_settings,
    retry,
    stage,
)
from lac_pipeline.trajectory import add_trajectory_columns, haversine
from lac_pipeline.transforms import merge_asof_by_icao
from lac_pipeline.validation import validate_pipeline_outputs


def main(argv: list[str] | None = None) -> int:
    run_lock = None
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

        # Obtain and format the date to retrieve data for (2 days ago by default)
        if settings.target_date:
            two_days_ago = datetime.strptime(settings.target_date, "%Y-%m-%d")
        else:
            two_days_ago = datetime.now() - timedelta(days=2)
        date_string = two_days_ago.strftime("%Y-%m-%d")
        logger.info("target_date date=%s", date_string)
        start = two_days_ago.replace(hour=0, minute=0, second=0, microsecond=0)
        end = two_days_ago.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_time = int(start.timestamp())
        start_hour = start_time - (start_time % 3600)
        end_time = int(end.timestamp())
        end_hour = end_time - (end_time % 3600)

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
        svdata4_query = (
            f"SELECT * FROM state_vectors_data4"
            f" WHERE icao24 LIKE '%'"
            f" AND time >= {start_time} AND time <= {end_time}"
            f" AND hour >= {start_hour} AND hour <= {end_hour}"
            f" AND lat >= {LAT_MIN} AND lat <= {LAT_MAX}"
            f" AND lon>= {LON_MIN} AND lon <= {LON_MAX}"
            f" AND geoaltitude >= {ALT_MIN} AND geoaltitude <= {ALT_MAX}"
            f" ORDER BY time"
        )

        with stage(logger, "query_state_vectors"):
            svdata4_df = retry(
                "state_vectors_data4",
                settings.query_attempts,
                settings.query_retry_delay_seconds,
                logger,
                lambda: Trino().query(
                    svdata4_query,
                    cached=False,
                    compress=True,
                ),
            )
        logger.info("query_rows table=state_vectors_data4 rows=%s", len(svdata4_df))

        if svdata4_df.empty:
            logger.warning("empty_source_day date=%s", date_string)
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
        icao24_str = ", ".join(f"'{item}'" for item in icao_list)

        ops_sts_query = (
            "SELECT icao24, mintime, maxtime, nacv, systemdesignassurance, version, "
            "positionnac, geometricverticalaccuracy, sourceintegritylevel, "
            "barometricaltitudeintegritycode FROM operational_status_data4"
            f" WHERE icao24 IN ({icao24_str})"
            f" AND mintime >= {start_time} AND maxtime <= {end_time}"
            f" AND hour >= {start_hour} AND hour <= {end_hour}"
            f" ORDER by mintime"
        )

        with stage(logger, "query_operational_status"):
            ops_sts_df = retry(
                "operational_status_data4",
                settings.query_attempts,
                settings.query_retry_delay_seconds,
                logger,
                lambda: Trino().query(
                    ops_sts_query,
                    cached=False,
                ),
            )
        logger.info("query_rows table=operational_status_data4 rows=%s", len(ops_sts_df))

        ops_sts_df["time"] = ops_sts_df["mintime"].astype("int64")

        # Save ops_sts pickle
        ops_sts_df.to_pickle(f"./OSN_pickles/opsstsdf_new_{date_string}.pkl")

        merged_df = merge_asof_by_icao(svdata4_df, ops_sts_df)

        # Third Query for Position data (to get the NIC)
        posdata4_query = (
            f"SELECT mintime, icao24, nic  FROM position_data4"
            f" WHERE icao24 IN ({icao24_str})"
            f" AND lat >= {LAT_MIN} AND lat <= {LAT_MAX}"
            f" AND lon>= {LON_MIN} AND lon <= {LON_MAX}"
            f" AND mintime >= {start_time} AND maxtime <= {end_time}"
            f" AND hour >= {start_hour} AND hour <= {end_hour}"
            f" ORDER by mintime"
        )

        with stage(logger, "query_position_data"):
            posdata4_df = retry(
                "position_data4",
                settings.query_attempts,
                settings.query_retry_delay_seconds,
                logger,
                lambda: Trino().query(
                    posdata4_query,
                    cached=False,
                ),
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

        if settings.dry_run:
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
    finally:
        if run_lock is not None:
            run_lock.release()
