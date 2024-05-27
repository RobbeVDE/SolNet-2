import pandas as pd
import numpy as np
from pvlib import location
import os
installation_int = 3
metadata = pd.read_pickle("Data/Sites/metadata.pkl")
latitude = metadata.loc[installation_int, "Latitude"]
longitude = metadata.loc[installation_int, "Longitude"]

total_df = pd.DataFrame()
print(os.getcwd())
for i in range(1,3):
    file = f"UK/CEDA_data_UK{i}.pickle"
    df = pd.read_pickle(file)
    total_df = pd.concat([total_df, df])
total_df.index = pd.to_datetime(total_df.index, utc=True)
print(total_df)
total_df["wind_speed_10m"] = np.sqrt(total_df["wind_u_10m"] **2+ total_df["wind_v_10m"] **2)
total_df["wind_direction_10m"] = np.arctan2(total_df["wind_v_10m"], total_df["wind_u_10m"]) *180/np.pi +180 #Convert from [-pi, pi] to [0,360]
total_df.drop(columns=['wind_u_10m', 'wind_v_10m'], inplace=True)
total_df['pressure_MSL'] = total_df['pressure_MSL']/100

#2016-07-14 has empty hour but didn't give an error so fill up with NaN values here, keep it general if more days missing
missing_days = [pd.to_datetime("2016-07-14 00:00", utc=True)]
for day in missing_days:
    hour_range = pd.date_range(day, day+pd.Timedelta('0h', tz="UTC"), freq='h')
    lol = pd.DataFrame(index=hour_range, columns=total_df.columns)
    total_df = pd.concat([total_df, lol])

total_df.sort_index(inplace=True)

print(total_df["2017-02-01":"2017-02-28"])
#Check if there are no dates missing now anymore
day_range = pd.date_range("2016-05-01", "2021-12-31", freq='D')
total_df['date'] = total_df.index.date
lol = day_range[~day_range.isin(total_df['date'])]
print(lol)
#Check for duplicates
print(total_df.index.duplicated(keep='first').sum())


#Reindex such that we have a value for 00:00, not that important bcs power=0
range = pd.date_range(total_df.index.min()-pd.Timedelta('1h'), total_df.index.max(), freq='h', tz="UTC")
total_df = total_df.reindex(index=range, method='nearest', limit=1)
# print(total_df.info())

#Finally, remove days with NaN, reason we added NaN is such that reindex not includes the whole day
mask = total_df.isna().any(axis=1) #Don't have to group on day bcs files were for day so if 1 hour is missing, all hours from that day are missing

total_df = total_df[~mask]
total_df.drop('date', inplace=True, axis=1)
# print(total_df["2017-02-01":"2017-02-28"])
print(total_df.info())
print(total_df)


# Only GHI given so use pvlib to calculate GDI and DNI

site = location.Location(latitude, longitude, tz='UTC')
times = total_df.index
solar_pos = site.get_solarposition(times)
zenith = np.deg2rad(solar_pos['apparent_zenith'])
# from pvlib import irradiance
# irr = irradiance.complete_irradiance(solar_pos["zenith"], ghi = total_df["downward_surface_SW_flux"], dhi = total_df["diffuse_surface_SW_flux"], dni=None)
# mask = irr['dni'].isnull()
# print(irr[mask].index)
# irr = irr.fillna(0)
# total_df.loc[mask, "diffuse_surface_SW_flux"] = total_df.loc[mask,"downward_surface_SW_flux"]
# total_df["direct_surface_SW_flux"] = irr["dni"]

total_df["direct_surface_SW_flux"] = total_df["direct_surface_SW_flux"].div(np.cos(zenith), axis=0, fill_value=0)


total_df.to_pickle("Data/Sites/NWP_3.pkl")