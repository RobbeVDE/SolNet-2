from ftplib import FTP
import os
import xarray as xr
import time
import pandas as pd
import numpy as np
# Important libraries
import os
from dotenv import load_dotenv
# Load secret .env file
load_dotenv()
installation_int = 1
metadata = pd.read_pickle("Data/Sites/metadata.pkl")
latitude = []
longitude = []
total_df_list = []
for i in range(4,9):
    # metadata = metadata.iloc[i]
 
    latitude.append(metadata.loc[i,'Latitude'])
    longitude.append(metadata.loc[i,'Longitude'])
    total_df_list.append(pd.DataFrame())

variables = ["temperature_1_5m", "relative_humidity_1_5m","total_cloud_amount","wind_u_10m","wind_v_10m", "downward_surface_SW_flux", "direct_surface_SW_flux", "diffuse_surface_SW_flux", "pressure_MSL"]



# Define the local directory name to put data in
ddir="C:/users/Robbe/SolNet-2"  #/users/students/r0778797

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
        # if (year==2019) and (month==2):
        #     start_day = 18
        # loop through days
        for day in range(start_day,ndays+1):
            day_df_list = [pd.DataFrame()]*5
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
                    start_timing = time.time()
                    f.retrbinary("RETR %s" % file, open(file, "wb").write)
                    df = xr.load_dataset(file, engine='cfgrib')
                    df.load()
                    var_df_list = []
                    for i in range(len(latitude)):

                        var_df = df.sel(latitude=latitude[i],longitude= longitude[i], method="nearest").to_dataframe()                
                        #make dataframe
                        var_df.set_index("valid_time", inplace=True)
                        end_day = int(day)+1
                        pred_time = f"{year}-{month}-{day}"
                        var_df = var_df[pred_time:pred_time]
                        var_df = var_df.iloc[:,-1] #last column is actual variable, others are mjeh
                        var_df.name = var
                        var_df_list.append(var_df)

                    os.remove(file)
                    try:
                        os.remove(f'{file}.9093e.idx') # Some sort of database file that Windows automatically generates
                    except:
                        pass
                    print(f'-----{time.time()-start_timing} seconds----')
                except:
                    print(f"Not able to retriece the data for {var} ar {day}/{month}/{year}")
                    df_dict = {var: [np.nan]*23}
                    index= pd.date_range(f"{year}-{month}-{day} 01:00", f"{year}-{month}-{day} 23:00", freq='h')
                    df = pd.DataFrame(df_dict, index=index)
                    var_df_list = [df]*5

                # Make month and day integers again
                for i in range(len(latitude)):
                    day_df_list[i] = day_df_list[i].join(var_df_list[i], how="right")
                day = int(day)
                month = int(month)
                print(f"Currently at {day}/{month}/{year}")
            
            #Merge day to total dataframe
            for i in range(len(latitude)):
                total_df_list[i] = pd.concat([total_df_list[i], day_df_list[i]])
                total_df_list[i].to_pickle(f"NL_{3+i}/CEDA_data_NL.pickle")
    
        
f.close()
