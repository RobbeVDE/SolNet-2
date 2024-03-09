from ftplib import FTP
import os
import xarray as xr
import time
start_time = time.time()
latitude = 51.0
longitude = 5.54
variables = ["temperature_1_5m", "relative_humidity_1_5m","total_cloud_amount","wind_u_10m","wind_v_10m","diffuse_surface_SW_flux","direct_surface_SW_flux", "downward_surface_SW_flux", "pressure_MSL"]

# Define the local directory name to put data in
ddir="C:/Users/Robbe/PycharmProjects/SolNet 2/Data"

# If directory doesn't exist make it
if not os.path.isdir(ddir):
   os.mkdir(ddir)

# Change the local directory to where you want to put the data
os.chdir(ddir)

f=FTP("ftp.ceda.ac.uk", "vrobbe", "3<[@$|RJLKF-")
# loop through years
for year in range(2016,2020):
    if year == 2016: #Only some months present
        month_range = range(5,13)
    else:
        month_range = range(1,13)
    # loop through months
    for month in month_range:
        # get number of days in the month
        if year%4==0 and month==2:
            ndays=29
        else:
            ndays=int("dummy 31 28 31 30 31 30 31 31 30 31 30 31".split()[month])

        # loop through days
        for day in range(1, ndays+1):

                # loop through variables
            for var in variables:
                f.cwd(f"/badc/ukmo-nwp/data/euro4-grib/{year}/{month}/{day}")
                # define filename
                file=f"{year}{month}{day}00_WSEuro4_MS20_ICAO_height_001054.grib"
                # get the remote file to the local directory
                f.retrbinary("RETR %s" % file, open(file, "wb").write)
                ds = xr.load_dataset(file, engine='cfgrib')
                df = ds.get("unknown")
                df = df.sel(latitude=latitude,longitude= longitude, method="nearest").to_dataframe()
                os.remove(file)
                os.remove(f'{file}.923a8.idx') # Some sort of database file that Windows automatically generates
                print(df)
                print(df.info())
                print(f'-----{time.time()-start_time} seconds----')
f.close()