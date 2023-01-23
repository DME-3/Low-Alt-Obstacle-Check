import datetime
import paramiko
import json
import pickle

### Constants
#
start = datetime.datetime(2022, 12, 22, 6, 0, 0)
end = datetime.datetime(2023, 1, 22, 23, 59, 59)

date_range = '%s_%s'% (str(start.date()), str(end.date()))
output_filename_pickle = 'OSN_pickles/svdata4_%s.pkl'%(date_range)

start_time = int(start.timestamp())
start_hour = start_time - (start_time % 3600)
end_time = int(end.timestamp())
end_hour = end_time - (end_time % 3600)

OSN_secrets_json = './OSN_secrets.json'

with open(OSN_secrets_json) as OSN_secrets:
    OSN_creds = json.load(OSN_secrets)

OSN_pwd = OSN_creds['OSN_password']
OSN_usr = OSN_creds['OSN_user']

callsign = "%"
icao24 = "%"

lat_min, lat_max = 50.896393, 50.967115
lon_min, lon_max = 6.919968, 7.005756
alt_min, alt_max = 0, 700

# Manage callsign exceptions for government flights
chx = "%CHX%"
hummel = "%HUMMEL%"
bpo = "%BPO%"
sar = "%SAR"
joker = "%JOKER%"
fck = "%FCK%"
ibis = "%IBIS%"
heli = "%HELI%"
airesc = "%AIRESC%"
gam = "%GAM%"
resq = "%RESQ%"

request = (
    f"-q select * from state_vectors_data4"
    f" where callsign like '{callsign}'"
    f" and callsign not like '{chx}'"
    f" and callsign not like '{hummel}'"
    f" and callsign not like '{bpo}'"
    f" and callsign not like '{sar}'"
    f" and callsign not like '{joker}'"
    f" and callsign not like '{fck}'"
    f" and callsign not like '{ibis}'"
    f" and callsign not like '{heli}'"
    f" and callsign not like '{airesc}'"
    f" and callsign not like '{gam}'"
    f" and callsign not like '{resq}'"
    f" and icao24 like '{icao24}'"
    f" and time>={start_time} and time<={end_time}"
    f" and hour>={start_hour} and hour<={end_hour}"
    f" and lat>={lat_min} and lat<={lat_max}"
    f" and lon>={lon_min} and lon<={lon_max}"
    f" and geoaltitude>={alt_min} and geoaltitude<={alt_max}"
)

###

def get_line_lst(line_txt):
    lst = line_txt.split("|")
    filter_lst = [elt.strip() for elt in lst]
    while filter_lst[0] == "":
        filter_lst = filter_lst[1:]
    while filter_lst[-1] == "":
        filter_lst = filter_lst[:-1]
    return filter_lst

#### Main program

# Connect to OSN and fetch data
p = paramiko.SSHClient()
p.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print('Connecting to OSN database...')
p.connect("data.opensky-network.org", port=2230, username=OSN_usr, password=OSN_pwd)
stdin, stdout, stderr = p.exec_command(request)
opt = stdout.readlines()
print('OSN data retrieved. Saving file...')
# Saving query result to pickle file
with open(output_filename_pickle, 'wb') as f:
  pickle.dump(opt, f)
f.close()
print('Finished')