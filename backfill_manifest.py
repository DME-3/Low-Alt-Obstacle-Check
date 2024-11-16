from sqlalchemy import create_engine, MetaData, insert
from datetime import datetime, timedelta
from sshtunnel import SSHTunnelForwarder
import logging
import json
import paramiko

# Enable SQLAlchemy logging for debugging (optional)
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

ed25519_key = paramiko.Ed25519Key(filename="./.ssh/id_ed25519")

MYSQL_secrets_json = "./mysql_secrets.json"

with open(MYSQL_secrets_json) as MYSQL_secrets:
    MYSQL_creds = json.load(MYSQL_secrets)

# Define the function to backfill the manifest table
def backfill_manifest(engine, start_date, end_date):
    """
    Backfills the manifest table with entries for each day between start_date and end_date.
    
    :param engine: SQLAlchemy engine object
    :param start_date: Start date for backfilling (datetime object)
    :param end_date: End date for backfilling (datetime object)
    """
    metadata = MetaData()
    metadata.reflect(bind=engine)
    manifest_table = metadata.tables['manifest']

    # Define the list of tables to backfill
    table_names = ['main_data', 'inf_data', 'gndinf_data']

    # Iterate through each day in the date range
    current_date = start_date
    while current_date <= end_date:
        for table_name in table_names:
            try:
                # Default values for the backfill entries
                processed_date = current_date.date()
                record_count = 0
                start_time = datetime.now()
                end_time = datetime.now()
                duration_sec = 0
                status = 'SUCCESS'
                error_message = None

                # Prepare the insert statement
                insert_stmt = insert(manifest_table).values(
                    table_name=table_name,
                    processed_date=processed_date,
                    record_count=record_count,
                    start_time=start_time,
                    end_time=end_time,
                    duration_sec=duration_sec,
                    status=status,
                    error_message=error_message
                )

                # Execute the insert statement
                with engine.begin() as connection:
                    connection.execute(insert_stmt)

                print(f"Backfilled manifest entry for table: {table_name}, date: {processed_date}")

            except Exception as e:
                print(f"Failed to backfill manifest entry for table: {table_name}, date: {processed_date}. Error: {e}")

        # Move to the next day
        current_date += timedelta(days=1)


with SSHTunnelForwarder(
    (MYSQL_creds["SSH_ADDRESS"], 22),
    ssh_username=MYSQL_creds["SSH_USERNAME"],
    ssh_pkey=ed25519_key,  # Use the loaded RSA key
    remote_bind_address=(
        MYSQL_creds["REMOTE_BIND_ADDRESS"],
        MYSQL_creds["REMOTE_BIND_PORT"],
    ),
    allow_agent=False,
) as tunnel:
    engstr = (
        "mysql+pymysql://"
        + MYSQL_creds["SSH_USERNAME"]
        + ":"
        + MYSQL_creds["PYANYWHERE_PASSWORD"]
        + "@127.0.0.1:"
        + str(tunnel.local_bind_port)
        + "/dme3$"
        + MYSQL_creds["PROD_DATABASE_NAME"]
    )

    engine = create_engine(engstr)

    # Define the date range for backfilling
    start_date = datetime.strptime('2021-01-02', '%Y-%m-%d')
    end_date = datetime.strptime('2024-11-14', '%Y-%m-%d')

    # Call the backfill function
    backfill_manifest(engine, start_date, end_date)
