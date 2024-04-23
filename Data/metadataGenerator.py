import pandas as pd
from Featurisation import target_renamer
import pyarrow.parquet as pq
"""
Sites:
0. NL 1 | 2. Costa Rica | 4.
1. NL 2 | 3. UK 1       | 5.

We want all azimuths in range [-180,180] where 0 means south
"""
n_sites = 4
columns = ["Name", "Installed Power", "Tilt", "Azimuth", "Latitude", "Longitude"]
metadata_df = pd.DataFrame(columns=columns, index=range(n_sites))
# 0.
installation_id = "3437BD60"
metadata = pd.read_csv("Data/installations Netherlands.csv", sep=';')
metadata = metadata.set_index('id')
metadata_id = metadata.loc[installation_id]
tilt = metadata_id["Tilt"]
peakPower = metadata_id["Watt Peak"]
azimuth = metadata_id["Orientation"]
latitude = metadata_id["Latitude"]
longitude = metadata_id["Longitude"]
prodNL = pd.read_parquet('Data/production.parquet', engine='pyarrow')
    
power = prodNL.loc[installation_id]
power = target_renamer(power, 'watt')
power = power.resample('h').mean()
power = power.tz_localize('UTC')

metadata_df.loc[0] = ["NL_1", peakPower, tilt, azimuth, latitude, longitude]
power.to_pickle("Data/Sites/PV_0.pkl")

# 1.
installation_id = "8723AX56"
metadata = pd.read_csv("Data/installations Netherlands.csv", sep=';')
metadata = metadata.set_index('id')
metadata_id = metadata.loc[installation_id]
tilt = metadata_id["Tilt"]
peakPower = metadata_id["Watt Peak"]
azimuth = metadata_id["Orientation"]
latitude = metadata_id["Latitude"]
longitude = metadata_id["Longitude"]

power = prodNL.loc[installation_id]
power = target_renamer(power, 'watt')
power = power.resample('h').mean()
power = power.tz_localize('UTC')

metadata_df.iloc[1,:] = ["NL_2", peakPower, tilt, azimuth, latitude, longitude]
power.to_pickle("Data/Sites/PV_1.pkl")

# 2.
peakPower = 5525
tilt = 8.5
azimuth = 90
latitude = 9.93676
longitude = -84.04388

metadata_df.iloc[2,:] = ["Costa Rica", peakPower, tilt, azimuth, latitude, longitude]
power = pd.read_pickle('Data/CostaRica.pkl')
power = target_renamer(power, 'Solar Production (W)')
power.to_pickle("Data/Sites/PV_2.pkl")

# 3.
installation_id = 26855
metadata = pd.read_csv("UK/metadata.csv", sep=',')
metadata = metadata.set_index('ss_id')
tilt = metadata.loc[installation_id,"tilt"]
peakPower = metadata.loc[installation_id,"kwp"]*1000
azimuth = metadata.loc[installation_id,"orientation"]-180 # To bring in desired range
latitude = metadata.loc[installation_id,"latitude_rounded"]
longitude = metadata.loc[installation_id,"longitude_rounded"]

total_df = pd.read_pickle("UK/ProdUK.pkl")

total_df['datetime'] = pd.to_datetime(total_df['datetime'])
total_df = total_df.set_index('datetime')
total_df = total_df.drop('ss_id', axis=1)
total_df = target_renamer(total_df, 'generation_wh')
total_df.to_pickle('Data/Sites/PV_3.pkl')
metadata_df.iloc[3,:] = ["UK", peakPower, tilt, azimuth, latitude, longitude]

print(metadata_df)

metadata_df.to_pickle("Data/Sites/metadata.pkl")