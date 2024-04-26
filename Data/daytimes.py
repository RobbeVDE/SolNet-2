import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

install_int = 0
metadata = pd.read_pickle("Data/Sites/metadata.pkl")
metadata = metadata.iloc[install_int]
lat = metadata['Latitude']
lon = metadata['Longitude']
# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": lat,
	"longitude": lon,
	"start_date": "2016-05-01",
	"end_date": "2022-12-31",
	"hourly": "is_day"
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
hourly_is_day = hourly.Variables(0).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["is_day"] = hourly_is_day

hourly_dataframe = pd.DataFrame(data = hourly_data)
hourly_dataframe.set_index('date', inplace=True)
print(hourly_dataframe["2016-05-01 03:00":"2016-05-01 09:00"])
hourly_dataframe.to_pickle(f'Data/Sites/is_day_{install_int}.pkl')
