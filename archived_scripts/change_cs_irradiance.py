from pvlib import irradiance, location
import pandas as pd
lat = 52
lon = 5

times = pd.date_range("2020-01-01","2021-01-01", freq='h', tz='UTC')
site = location.Location(lat,lon)
cs = site.get_clearsky(times)
cs_max = cs.groupby(cs.index.date).max()
print((cs_max.diff()/cs_max).max())