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

Event flags are added to the gdf dataframe, and two additional dataframes containing only events are created. They are then clean to apply lateral and vertical tolerances, and saved (by default in dataframes/ directory).  

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