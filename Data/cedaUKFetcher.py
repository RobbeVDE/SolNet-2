from ftplib import FTP
import os
import xarray as xr
import time
import numpy as np
import pandas as pd
# Important libraries
import os
from dotenv import load_dotenv
# Load secret .env file
load_dotenv()

latitude, longitude = 53.4,-2.91

variables = ["temperature_1_5m", "relative_humidity_1_5m","total_cloud_amount","wind_u_10m","wind_v_10m", "downward_surface_SW_flux", "direct_surface_SW_flux", "diffuse_surface_SW_flux", "pressure_MSL"]
total_df = pd.DataFrame()


# Define the local directory name to put data in
ddir="C:/Users/Robbe/PycharmProjects/SolNet 2/UK"

# If directory doesn't exist make it
if not os.path.isdir(ddir):
   os.mkdir(ddir)

# Change the local directory to where you want to put the data
os.chdir(ddir)

f=FTP("ftp.ceda.ac.uk", os.getenv('ceda_usrname'), os.getenv('ceda_pssword'))
# loop through years
for year in range(2016,2022):
    if year == 2016: #Only some months present
        month_range = range(5,13)
    else:
        month_range = range(1,13)
    # loop through months
    for month in month_range:
        # get number of days in the month
        start_day = 1
        if year%4==0 and month==2:
            ndays=29
        else:
            ndays=int("dummy 31 28 31 30 31 30 31 31 30 31 30 31".split()[month])
        if (year==2016) and (month==3):
            start_day = 1
        # loop through days
        for day in range(start_day,ndays+1):
            day_df = pd.DataFrame()
                # loop through variables
            for var in variables:
                #Correct the numbers <10 to have format like 05 ipv 5:
                if day < 10:
                    day = "0" + str(day)
                if month < 10:
                    month = "0" + str(month)
                try:
                    f.cwd(f"/badc/ukmo-nwp/data/euro4-grib/{year}/{month}/{day}")
                    # define filename
                    file=f"{year}{month}{day}00_WSEuro4_{var}_001054.grib"
                    # get the remote file to the local directory
                    t0 = time.time()
                    f.retrbinary("RETR %s" % file, open(file, "wb").write)
                    df = xr.load_dataset(file, engine='cfgrib')
                    df.load()
                    df = df.sel(latitude=latitude,longitude= longitude, method="nearest").to_dataframe()
                    os.remove(file)
                    os.remove(f'{file}.923a8.idx') # Some sort of database file that Windows automatically generates
                except:
                    print(f"Not able to retriece the data for {var} ar {day}/{month}/{year}")
                    df_dict = {var: [np.nan]*23}
                    index= pd.date_range(f"{year}-{month}-{day} 01:00", f"{year}-{month}-{day} 23:00", freq='h')
                    df = pd.DataFrame(df_dict, index=index)

                #make dataframe
                df.set_index("valid_time", inplace=True)
                end_day = int(day)+1
                pred_time = f"{year}-{month}-{day}"
                df = df[pred_time:pred_time]
                df = df.iloc[:,-1] #last column is actual variable, others are mjeh
                df.name = var
                day_df = day_df.join(df, how="right")
                print(f'-----{time.time()-t0} seconds----')

                # Make month and day integers again
                day = int(day)
                month = int(month)
            
            #Merge day to total dataframe
            total_df = pd.concat([total_df, day_df])
            total_df.to_pickle("CEDA_data_UK.pickle")
            print(f"Currently at {day}/{month}/{year}")
    total_df.to_pickle(f"CEDA_backup_{year}")
        
f.close()