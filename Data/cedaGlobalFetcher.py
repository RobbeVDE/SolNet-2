from ftplib import FTP
import os
import xarray as xr
import time
import pandas as pd
# Important libraries
import os
from dotenv import load_dotenv
# Load secret .env file
load_dotenv()

latitude = 9.93676
longitude = -84.04388+360 #+360 bcs it is in range [0,360] but not for are A then it is [-45,45]
area='D' #You should change this according to latitude and longitude
variables = ["temperature_1_5m", "relative_humidity_1_5m","total_cloud","wind_u_10m","wind_v_10m", "Total_Downward_Surface_SW_Flux", "mean_sea_level_pressure"]
total_df = pd.DataFrame()


# Define the local directory name to put data in
ddir="C:/Users/Robbe/PycharmProjects/SolNet 2/CostaRica"

# If directory doesn't exist make it
if not os.path.isdir(ddir):
   os.mkdir(ddir)

# Change the local directory to where you want to put the data
os.chdir(ddir)

f=FTP("ftp.ceda.ac.uk", os.getenv('ceda_usrname'), os.getenv('ceda_pssword'))
# loop through years
for year in range(2016,2021):
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
                f.cwd(f"/badc/ukmo-nwp/data/global-grib/{year}/{month}/{day}")
                # define filename
                file=f"{year}{month}{day}00_WSGlobal17km_{var}_Area{area}_000144.grib"
                # get the remote file to the local directory
                start_timing = time.time()
                f.retrbinary("RETR %s" % file, open(file, "wb").write)
                df = xr.load_dataset(file, engine='cfgrib')
                df.load()
                df = df.sel(latitude=latitude,longitude= longitude, method="nearest").to_dataframe()
                os.remove(file)
                os.remove(f'{file}.923a8.idx') # Some sort of database file that Windows automatically generates

                #make dataframe
                df.set_index("valid_time", inplace=True)
                end_day = int(day)+1
                pred_time = f"{year}-{month}-{day}"
                df = df[pred_time:pred_time]
                print(df)
                df = df.iloc[:,-1] #last column is actual variable, others are mjeh
                df.name = var
                day_df = day_df.join(df, how="right")
                print(day_df)
                print(f'-----{time.time()-start_timing} seconds----')

                # Make month and day integers again
                day = int(day)
                month = int(month)
            
            #Merge day to total dataframe
            total_df = pd.concat([total_df, day_df])
            total_df.to_pickle("CEDA_data_CR.pickle")
    total_df.to_pickle(f"CEDA_backup_{year}")
        
f.close()