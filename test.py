import xarray as xr
ds = xr.load_dataset("Data/testCEDA.grib", engine='cfgrib')
for v in ds:
    key = str(v)
print(ds)
df = ds
latitude = 51.0
longitude = 5.54
df = df.sel(latitude=latitude,longitude= longitude, method="nearest").to_dataframe()
print(df)
print(df.info())
