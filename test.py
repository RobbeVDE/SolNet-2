from pvlib import location
import pandas as pd
import numpy as np
total_df = pd.read_pickle("CEDA_dataNL.pickle")

latitude,longitude = 52.0499, 5.07391
site = location.Location(latitude, longitude, tz='UTC')
times = total_df.index
solar_pos = site.get_solarposition(times)
zenith = np.deg2rad(solar_pos['apparent_zenith'])

dir_surf_irrad = total_df["direct_surface_SW_flux"]
print(dir_surf_irrad)
print(dir_surf_irrad.div(np.cos(zenith, dtype='float64'), axis=1, fill_value=0))



total_df["direct_surface_SW_flux"] = dir_surf_irrad.div(np.cos(zenith, dtype='float64'), axis=0, fill_value=0)