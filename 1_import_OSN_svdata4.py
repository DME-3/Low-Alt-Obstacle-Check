from datetime import datetime
from pyopensky.trino import Trino
import sys

LAT_MIN, LAT_MAX = 50.896393, 50.967115
LON_MIN, LON_MAX = 6.919968, 7.005756
ALT_MIN, ALT_MAX = 0, 750 # update from 700 m to 750 m, in line with CTR limit at 2500 ft plus margin (and accounting for Geoid Height)

TIMEOUT = 30 # timeout for the connection to OSN Impala shell (in seconds)

def get_dates():
    try:
        start_date_str = input("Enter start date in DD/MM/YY format: ")
        end_date_str = input("Enter end date in DD/MM/YY format: ")
        start_date = datetime.strptime(start_date_str, '%d/%m/%y')
        end_date = datetime.strptime(end_date_str, '%d/%m/%y')

        # Modify start_date to be 00:00:00
        start_date = start_date.replace(hour=0, minute=0, second=1, microsecond=0)
        # Modify end_date to be 23:59:59
        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    except ValueError:
        print('Incorrect date format, should be DD/MM/YY')
        sys.exit(1)
    
    return start_date, end_date

def setup_query(start, end):

    start_time = int(start.timestamp())
    start_hour = start_time - (start_time % 3600)
    end_time = int(end.timestamp())
    end_hour = end_time - (end_time % 3600)

    callsign = "%"
    icao24 = "%"

    query = (
        f"SELECT * FROM state_vectors_data4"
        f" WHERE callsign LIKE '{callsign}'"
        f" AND icao24 LIKE '{icao24}'"
        f" AND time >= {start_time} AND time <= {end_time}"
        f" AND hour >= {start_hour} AND hour <= {end_hour}"
        f" AND lat >= {LAT_MIN} AND lat <= {LAT_MAX}"
        f" AND lon>= {LON_MIN} AND lon <= {LON_MAX}"
        f" AND geoaltitude >= {ALT_MIN} AND geoaltitude <= {ALT_MAX}"
    )

# needed: state_vectors_data4.time, state_vectors_data4.icao24, state_vectors_data4.lat, state_vectors_data4.lon, state_vectors_data4.callsign, state_vectors_data4.onground, state_vectors_data4.baroaltitude, state_vectors_data4.geoaltitude, state_vectors_data4.lastposupdate, state_vectors_data4.lastcontact, state_vectors_data4.hour

    return query

# Connect to OSN and fetch data
def get_OSN_svdata4(start, end):

    query = setup_query(start, end)

    trino = Trino()

    df = trino.query(
        query,
        cached=False,
        compress=True,
    )

    return df

if __name__ == "__main__":

    print('Please enter start and end dates for the OSN query:')
    start, end = get_dates()

    date_range = '%s_%s'% (str(start.date()), str(end.date()))
    output_filename_pickle = 'OSN_pickles/svdata4df_%s.pkl'%(date_range)

    input("Press enter to continue and request OSN data for date range %s"%(date_range))

    osn_data = get_OSN_svdata4(start, end)
    print('OSN data retrieved.')

    osn_data = osn_data.drop('serials', axis = 1) # drop the serials columns, as it contains lists it is non-hashable and therefore the .duplicated() would not work

    duplicates = osn_data.duplicated()
    num_duplicates = duplicates.sum()
    print(f'Number of duplicate rows to remove: {num_duplicates}')

    osn_data_dedup = osn_data.drop_duplicates()
    osn_data_dedup.reset_index(drop=True, inplace=True)

    # Saving query result to pickle file
    print('Saving file...')
    osn_data_dedup.to_pickle(output_filename_pickle)

    print('Finished')