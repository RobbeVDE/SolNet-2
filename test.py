import xarray as xr

print(xr.load_dataset("test.grib", engine='cfgrib'))