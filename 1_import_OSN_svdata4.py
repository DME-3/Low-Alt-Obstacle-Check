from datetime import datetime
import paramiko
import json
import pickle
import sys

OSN_secrets_json = './OSN_secrets.json'

LAT_MIN, LAT_MAX = 50.896393, 50.967115
LON_MIN, LON_MAX = 6.919968, 7.005756
ALT_MIN, ALT_MAX = 0, 700

TIMEOUT = 30 # timeout for the connection to OSN Impala shell (in seconds)

# Callsign exceptions for government, military and ambulance flights
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

def setup_request(start, end):

    start_time = int(start.timestamp())
    start_hour = start_time - (start_time % 3600)
    end_time = int(end.timestamp())
    end_hour = end_time - (end_time % 3600)

    callsign = "%"
    icao24 = "%"

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
        f" and lat>={LAT_MIN} and lat<={LAT_MAX}"
        f" and lon>={LON_MIN} and lon<={LON_MAX}"
        f" and geoaltitude>={ALT_MIN} and geoaltitude<={ALT_MAX}"
    )

    return request

# Connect to OSN and fetch data
def get_OSN_svdata4(start, end):

    with open(OSN_secrets_json) as OSN_secrets:
        OSN_creds = json.load(OSN_secrets)

    OSN_pwd = OSN_creds['OSN_password']
    OSN_usr = OSN_creds['OSN_user']

    request = setup_request(start, end)

    opt=[]

    p = paramiko.SSHClient()
    p.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print('Connecting to OSN database...')
    p.connect("data.opensky-network.org", port=2230, username=OSN_usr, password=OSN_pwd, timeout=TIMEOUT)
    stdin, stdout, stderr = p.exec_command(request)

    while True:
        line = stdout.readline()
        if not line:
            break
        #print(line, end="")
        opt.append(line)

    return opt

if __name__ == "__main__":

    print('Please enter start and end dates for the OSN query:')
    start, end = get_dates()

    date_range = '%s_%s'% (str(start.date()), str(end.date()))
    output_filename_pickle = 'OSN_pickles/svdata4_%s.pkl'%(date_range)

    input("Press enter to continue and request OSN data for date range %s"%(date_range))

    osn_data = get_OSN_svdata4(start, end)

    print('OSN data retrieved. Saving file...')
    # Saving query result to pickle file
    with open(output_filename_pickle, 'wb') as f:
        pickle.dump(osn_data, f)
    
    f.close()
    print('Finished')