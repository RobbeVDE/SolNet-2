import pandas as pd
import numpy as np
from pvlib import location

for nl_int in range(3,8):
    metadata = pd.read_pickle("Data/Sites/metadata.pkl")
    latitude = metadata.loc[nl_int+1, "Latitude"]
    longitude = metadata.loc[nl_int+1, "Longitude"]
    

    total_df = pd.read_pickle(f"NL_{nl_int}/CEDA_data_NL.pickle")
    total_df.index = pd.to_datetime(total_df.index, utc=True)
    total_df["wind_speed_10m"] = np.sqrt(total_df["wind_u_10m"] **2+ total_df["wind_v_10m"] **2)
    total_df["wind_direction_10m"] = np.arctan2(total_df["wind_u_10m"], total_df["wind_v_10m"]) *180/np.pi +180  #Convert from [-pi, pi] to [0,360]
    total_df.drop(columns=['wind_u_10m', 'wind_v_10m'], inplace=True)
    total_df['pressure_MSL'] = total_df['pressure_MSL']/100

    print(total_df)


    # #2016-07-14 has empty columns but didn't give an error so fill up with NaN values here, keep it general if more days missing
    # missing_days = [pd.to_datetime("2016-07-14", utc=True)]
    # for day in missing_days:
    #     hour_range = pd.date_range(day, day+pd.Timedelta('23h', tz="UTC"), freq='h')
    #     lol = pd.DataFrame(index=hour_range, columns=total_df.columns)
    #     total_df = pd.concat([total_df, lol])

    total_df.sort_index(inplace=True)

    print(total_df["2017-02-01":"2017-02-28"])
    #Check if there are no dates missing now anymore
    day_range = pd.date_range("2017-05-01", "2021-12-31", freq='D')
    total_df['date'] = total_df.index.date
    lol = day_range[~day_range.isin(total_df['date'])]
    print(lol)
    #Check for duplicates
    print(total_df[total_df.index.duplicated(keep='first')].index)
    total_df = total_df.drop_duplicates()

    #Reindex such that we have a value for 00:00, not that important bcs power=0
    range = pd.date_range(total_df.index.min()-pd.Timedelta('1h'), total_df.index.max(), freq='h', tz="UTC")
    total_df = total_df.reindex(index=range, method='nearest', limit=1)
    # print(total_df.info())

    #Finally, remove days with NaN, reason we added NaN is such that reindex not includes the whole day
    mask = total_df.isna().any(axis=1) #Don't have to group on day bcs files were for day so if 1 hour is missing, all hours from that day are missing

    total_df = total_df[~mask]
    total_df.drop('date', inplace=True, axis=1)
    # print(total_df["2017-02-01":"2017-02-28"])
    # print(total_df.info())
    # print(total_df)


    # Direct irradiance is given in plane instead of normal we can however recalculate this with DNI = DirIrrad/cos(zenith)

    site = location.Location(latitude, longitude, tz='UTC')
    print(latitude, longitude)
    times = total_df.index
    solar_pos = site.get_solarposition(times)
    zenith = np.deg2rad(solar_pos['apparent_zenith'])

    from pvlib import irradiance
    irr = irradiance.complete_irradiance(solar_pos["zenith"], ghi = total_df["downward_surface_SW_flux"], dhi = total_df["diffuse_surface_SW_flux"], dni=None)
    mask = irr['dni'].isnull()
    print(irr[mask].index)
    irr = irr.fillna(0)
    total_df.loc[mask, "diffuse_surface_SW_flux"] = total_df.loc[mask,"downward_surface_SW_flux"]
    total_df["direct_surface_SW_flux"] = irr["dni"]


    total_df.to_pickle(f"Data/Sites/NWP_{nl_int+1}.pkl")