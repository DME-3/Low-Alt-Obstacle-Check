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
    lst = line_txt.split("|")
    filter_lst = [elt.strip() for elt in lst]
    while filter_lst[0] == "":
        filter_lst = filter_lst[1:]
    while filter_lst[-1] == "":
        filter_lst = filter_lst[:-1]
    return filter_lst

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
    opt = stdout.readlines()

    columns = [
        "icao24",
        "positionnac",
        "maxtime",
    ]

    lst_of_lst = []

    for i in range(3, len(opt) - 1):
        if opt[i][:2] != "+-":
            l = get_line_lst(opt[i])
            if len(l) != len(columns):
                print("Error in parsing line: ")
                print(l)
                print(len(l))
            lst_of_lst.append(l)

    nac_df = pd.DataFrame(lst_of_lst, columns=columns)
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

    if os.path.isfile(df_path + '/nac_df.json'):
        print('Error: nac_df.json already present in %s. Check if the dataframes have not been processed yet.'%(df_path))
        sys.exit(1)

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