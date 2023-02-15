import paramiko
import json
import pandas as pd
import random
import os
import sys
from glob import glob

OSN_secrets_json = './OSN_secrets.json'

NAC_MIN = 10 # 

with open(OSN_secrets_json) as OSN_secrets:
    OSN_creds = json.load(OSN_secrets)

OSN_pwd = OSN_creds['OSN_password']
OSN_usr = OSN_creds['OSN_user']

def get_line_lst(line_txt):
    '''
    Return a list of non-empty strings that were delimited by the "|" character in the input string.
    Leading and trailing whitespace are removed
    '''
    return [elt.strip() for elt in line_txt.split("|") if elt.strip()]

def fetch_data_from_OSN(gdf):

    nac_query = '-q '

    first = True

    all_flights = gdf.ref.unique()

    for flight in all_flights:

        flight_data = gdf[gdf['ref'] == flight]

        k = random.randint(0,len(flight_data)-1) # randomly choose a point for the current flight. It will be used to determine NACp for the flight. This is in order to simplify the Impala query
        flight_pt = flight_data.iloc[k]

        timestamp = flight_pt.time

        req =   ('SELECT icao24, positionnac, maxtime FROM operational_status_data4 WHERE '
                'hour=%s AND maxtime>%s-10 AND maxtime<%s+10 AND icao24 LIKE %s') \
                % (str(flight_pt.hour), str(timestamp), str(timestamp), '\'%'+flight_pt.icao24+'%\' LIMIT 1 ')

        if not(first): 
            nac_query += 'UNION '

        first = False

        nac_query += req

    print('Connecting to OSN database...')
    p = paramiko.SSHClient()
    p.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    p.connect("data.opensky-network.org", port=2230, username=OSN_usr, password=OSN_pwd)
    stdin, stdout, stderr = p.exec_command(nac_query)
    osn_data = stdout.readlines()

    columns = [
        "icao24",
        "positionnac",
        "maxtime",
    ]

    # Load OSN data in a dataframe
    nac_df = pd.DataFrame([get_line_lst(osn_data[i]) for i in range(3, len(osn_data) - 1) if not osn_data[i].startswith("+-")], columns=columns)

    nac_df.dropna(inplace=True)

    nac_df['positionnac'] = nac_df['positionnac'].apply(lambda x: int(x))

    return nac_df

if __name__ == "__main__":

    try:
        arg1 = sys.argv[1]
    except IndexError:
        print('Usage: ' + os.path.basename(__file__) + ' <path with dataframes json files to process>')
        sys.exit(1)

    df_path = arg1

    print('Loading dataframes...')

    # Load clean_inf_df
    inf_df_files = glob(df_path + '/clean_inf_df*.json')
    clean_inf_df = pd.concat([pd.read_json(file, lines=False) for file in inf_df_files])
    clean_inf_df = clean_inf_df.reset_index(drop=True)

    # Load clean_gnd_inf_df
    gnd_inf_df_files = glob(df_path + '/clean_gnd_inf_df*.json')
    clean_gnd_inf_df = pd.concat([pd.read_json(file, lines=False) for file in gnd_inf_df_files])
    clean_gnd_inf_df = clean_gnd_inf_df.reset_index(drop=True)

    # Load gdf
    gdf_files = glob(df_path + '/gdf*.json')
    gdf = pd.concat([pd.read_json(file, lines=False) for file in gdf_files])
    gdf = gdf.reset_index(drop=True)

    if os.path.isfile(df_path + '/nac_df.json'):
        print('nac_df.json already present in %s, loading it.'%(df_path))
        nac_df = pd.read_json(df_path + '/nac_df.json', lines=False)
        nac_df = nac_df.reset_index(drop=True)
    else:
        nac_df = fetch_data_from_OSN(gdf)
    
    ac_bad_nac = nac_df[nac_df['positionnac'] < NAC_MIN].icao24.unique()
    print('Data retrieved. Out of %s flights, %s flights have a NACp below %s'%(str(len(nac_df)), str(len(ac_bad_nac)), str(NAC_MIN)))

    #save the nac_df
    nac_json = df_path + '/nac_df.json'
    nac_df.to_json(nac_json)
    print('Saved %s'%(nac_json))

    print('Removing the flights from the dataframes...')
    index_to_drop = clean_inf_df[clean_inf_df['icao24'].isin(ac_bad_nac)].index
    clean_inf_df.drop(index_to_drop, inplace = True)
    gnd_index_to_drop = clean_gnd_inf_df[clean_gnd_inf_df['icao24'].isin(ac_bad_nac)].index
    clean_gnd_inf_df.drop(gnd_index_to_drop, inplace = True)
    gdf_index_to_drop = gdf[gdf['icao24'].isin(ac_bad_nac)].index
    gdf.drop(gdf_index_to_drop, inplace = True)

    print('Saving the pruned dataframes...')
    for gdf_file in gdf_files:
        gdf_json = gdf_file
        gdf.to_json(gdf_json)
        print('Saved %s'%(gdf_json))
    for inf_file in inf_df_files:
        inf_json = inf_file
        clean_inf_df.to_json(inf_json)
        print('Saved %s'%(inf_json))
    for gnd_inf_file in gnd_inf_df_files:
        gnd_inf_json = gnd_inf_file
        clean_gnd_inf_df.to_json(gnd_inf_json)
        print('Saved %s'%(gnd_inf_json))

    print('Done, exiting.')