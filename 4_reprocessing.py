# Filter out aircraft known to have NACp < 10 and for which no nac_df is available in the historical data (based on past processing)

from glob import glob
import pandas as pd
import sys
import os

# list for 2021-2022
list = ['DEEFQ', 'DEGZZ', 'DEISK', 'DEMCG', 'DEPTS', 'DEWIM', 'DEZEI',
       'DKIOG', 'DMCDS', 'DMEID', 'DMYMZ', 'DMZDY']

if __name__ == "__main__":
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print('Usage: ' + os.path.basename(__file__) + ' <.json file to process>')
        sys.exit(1)

    df_file = arg1

    df = pd.read_json(df_file, lines=False)
    df = df.reset_index(drop=True)

    index_to_drop = df[df['callsign'].isin(list)].index
    print('%s occurences to drop in file %s'%(str(len(index_to_drop)), df_file))
    df.drop(index_to_drop, inplace = True)

    df.to_json(df_file)
    print('Saved %s. Exiting.'%(df_file))