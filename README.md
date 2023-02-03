# Low-Alt-Obstacle-Check
  
## General  

These Python scripts are used to generate the data for the [Low Alt Cologne](https://www.lowaltcologne.org) website.  

The process is made of the 5 steps described below.  

For step 1, sources of data other than OSN may be used alternatively as long as the format is preserved.  

Shell scripts may be used to run the scripts iteratively if many files are processed.  

requirements.txt lists required libraries for all scripts.  

## 1 - Importing OSN data  

Requisite:  
- OSN_secrets.json file with following format:

```
{
    "OSN_user": "username",
    "OSN_password": "password"
}
```  
  
User inputs the appropriate start and end times for querying the OSN database.  

The script manages callsign exclusions for authorised low-altitude operations, such as HEMS, governmental, police, military...  

The output is a pickle file which is saved in the local 'OSN_pickles' directory.  

## 2 - Converting data and checking altitude  

Requisites:
- DEM file in resources/ directory
- Obstacles JSON file in resources/ directory
- polygons.py for city and Rhein polygons
- den_func.py for functions getting elevation from DEM

Run the script passing the input pickle as only argument.  

The script converts the pickle input in a global dataframe (gdf), adds the distance, time, ground elevation information, then calculate possible events with respect to obstacles and minimum ground heights.  

Obstacle events occur when an aircraft is within a 600 m (2000 ft) radius of an obstacle, at a height lower than the obstacle height plus 300 m (1000 ft). The aircraft must be over Cologne, and not over the Rhein river (not considered as congested area).  

Ground events occur when an aircraft flies at a height lower than 300 m (1000 ft) over ground.  

Event flags are added to the gdf dataframe, and two additional dataframes containing only events are created. They are then clean to apply lateral and vertical tolerances, and saved (by default in dataframes/ directory).  

![](https://github.com/DME-3/Low-Alt-Obstacle-Check/raw/main/LowAltCologne_Definitions.png)

## 3 - Getting NACp parameters and removing data points where NACp is insufficient  

Requisites:  
- OSN_secrets.json file, or
- nac_df file in the dataframes directories if already retrieved  

Run the script passing a directory containing dataframes as the only argument.

If a nac_df.json file is already present, the data will be used. If not, the script will connect to the OSN database and retrieve NACp parameter for each flight in the dataframes. This will be done only for one position point.  

The dataframes are then processed to remove any aircraft data where the NACp is insufficient (by default NACp < 10, i.e. 10 m horizontal accuracy), then saved.  

## 4 - Reprocessing

Run the script passing a dataframe .json file as the only argument.

The script contains a list of callsigns known to transmit data with insufficient accuracy, or authorised to fly at low altitude.  

These aircraft are removed from the dataframe, which is then saved.  

## 4bis - EU Rotors

Run the script passing a dataframe .json file as the only argument.

The script contains a list of callsigns and dates corresponding to the EU Rotors symposium taking place in Cologne. These aircraft have special authorisation to land in the city. They are removed from the dataframe, which is then saved.  

## Limitations

Known limitations:

- Some flights are split and appear as distinct flights in the data (distinct 'ref' identifiers), although the time difference does not exceed the detection threshold.
- Minimum ground altitude check is made with a default elevation value (50 m) instead of using gnd_elev parameter.
- Not all authorised aircraft (e.g. HELI955) may have been properly filtered for older data (2021).
- inf_pt and gnd_inf_pt flags in gdf will be set to True irrespective of margins. This may flag a flight in the site's map, which does not appear in the event list.
- infractions for bridges over the Rhein river are not calculated. 
- timestamp format changes at step 3 from string to datetime even if NACp list is empty.
