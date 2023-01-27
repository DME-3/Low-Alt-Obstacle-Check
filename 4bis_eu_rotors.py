# Filter out aircraft participating to the European Rotors Symposium in Cologne
# These aircraft have special authorisation to land in Cologne
# Set the ground flags to False for those aircraft at the symposium dates

import pandas as pd
import sys
import os
from datetime import datetime

rotors_dates = ['13/11/21 - 19/11/21', '05/11/22 - 11/11/22']
rotors_aircraft = ['OEXAU', 'OEXQE', 'DHXBA', 'CFNFO', 'HBZWE', 'DHUGO', 'DHHLJ', 'DHRAN', 'DHFEJ', 'T7BELL', 'DHXCB', 'FHUHU', 'LID703', 'GNHVI']

# Function to check if timestamp is within a date interval
def is_timestamp_in_interval(timestamp, date_interval):
    start, end = [datetime.strptime(date, '%d/%m/%y') for date in date_interval.split(' - ')]
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
    return start <= datetime.fromtimestamp(timestamp) <= end

def process_gdf(df_file):

    gdf = pd.read_json(df_file, lines=False)
    gdf = gdf.reset_index(drop=True)

    # Iterate over the list of date intervals
    for interval in rotors_dates:
        # Define a boolean mask based on the filtering conditions
        mask = (gdf['callsign'].isin(rotors_aircraft)) & (gdf['time'].apply(lambda x: is_timestamp_in_interval(x, interval)))

        # Set column 'A' to True if any value in column 'B' is True, False otherwise
        gdf.loc[mask, 'gnd_inf_flt'] = False
        gdf.loc[mask, 'gnd_inf_pt'] = False
    
    return gdf

def process_inf_df(df_file):

    df = pd.read_json(df_file, lines=False)
    df = df.reset_index(drop=True)

    # Iterate over the list of date intervals
    for interval in rotors_dates:
        # Define a boolean mask based on the filtering conditions
        mask = (df['callsign'].isin(rotors_aircraft)) & (df['hour'].apply(lambda x: is_timestamp_in_interval(x, interval)))

        df.drop(df[mask].index, inplace=True)

    return df

if __name__ == "__main__":
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print('Usage: ' + os.path.basename(__file__) + ' <.json file to process>')
        sys.exit(1)

    df_file = arg1

    if '/gdf_' in df_file:
        gdf = process_gdf(df_file)
        gdf.to_json(df_file)
        print('Processed and saved gdf file %s. Exiting.'%(df_file))
    elif '/clean_gnd_inf_' in df_file:
        gnd_inf_df = process_inf_df(df_file)
        gnd_inf_df.to_json(df_file)
        print('Processed and saved gnd_inf file %s. Exiting.'%(df_file))
    else:
        print('Skipping file %s'%(df_file))
    
    print('Done, exiting.')