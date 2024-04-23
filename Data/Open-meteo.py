import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry


install_int = 3
metadata = pd.read_pickle("Data/Sites/metadata.pkl")
metadata = metadata.iloc[install_int]
lat = metadata['Latitude']
lon = metadata['Longitude']
# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": lat,
	"longitude": lon,
	"start_date": "2016-05-01",
	"end_date": "2021-12-31",
	"hourly": ["temperature_2m", "relative_humidity_2m", "pressure_msl", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance"],
	"timezone": "GMT",
    "wind_speed_unit": "ms",
	"models": "era5_seamless"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_pressure_msl = hourly.Variables(2).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(4).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(5).ValuesAsNumpy()
hourly_shortwave_radiation = hourly.Variables(6).ValuesAsNumpy()
hourly_diffuse_radiation = hourly.Variables(7).ValuesAsNumpy()
hourly_direct_normal_irradiance = hourly.Variables(8).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["temperature_2m"] = hourly_temperature_2m + 273.15 #From Celcius to Kelvin
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["pressure_msl"] = hourly_pressure_msl
hourly_data["cloud_cover"] = hourly_cloud_cover/100
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
hourly_data["shortwave_radiation"] = hourly_shortwave_radiation
hourly_data["diffuse_radiation"] = hourly_diffuse_radiation
hourly_data["direct_normal_irradiance"] = hourly_direct_normal_irradiance

hourly_dataframe = pd.DataFrame(data = hourly_data)
hourly_dataframe.set_index('date', inplace=True)
hourly_dataframe.to_pickle(f'Data/Sites/Reanalysis_{install_int}.pickle')
print(hourly_dataframe)