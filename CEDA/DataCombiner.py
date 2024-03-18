import pandas as pd
import numpy as np
total_df = pd.DataFrame()
for i in range(1,12):
    file = f"CEDA/CEDA_data{i}.pickle"
    df = pd.read_pickle(file)
    total_df = pd.concat([total_df, df])
total_df.index = pd.to_datetime(total_df.index, utc=True)
total_df["wind_speed_10m"] = total_df["wind_u_10m"] **2+ total_df["wind_v_10m"] **2
total_df["wind_direction_10m"] = np.arctan2(total_df["wind_v_10m"], total_df["wind_u_10m"]) *180/np.pi + 180 #Convert from [-pi, pi] to [0,360]
total_df.drop(columns=['wind_u_10m', 'wind_v_10m'], inplace=True)
total_df.to_pickle("CEDA_data.pickle")