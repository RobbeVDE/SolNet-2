import xarray as xr
ds = xr.load_dataset("Data/testCEDA.grib", engine='cfgrib')
for v in ds:
    print("{}, {}, {}".format(v, ds[v].attrs["long_name"], ds[v].attrs["units"]))

df = ds.get("unknown")
latitude = 51.0
longitude = 5.54
df = df.sel(latitude=latitude,longitude= longitude, method="nearest").to_dataframe()
print(df)
print(df.info())
