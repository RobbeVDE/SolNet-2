import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Data.Featurisation import Featurisation
from main import target_renamer, forecast_maker, data_slicer
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


#This experiment is done to explain the shaky behaviour of the neural network
# Linear = bad --> check data again
# Linear = good --> try with tensorisation
#Then bad, check tensorisation again



month_data = True #1 month of data for testing new code

target_month = pd.date_range("2019-08-01", "2019-08-31 23:00", freq='h', tz="UTC")
eval_month = pd.date_range("2019-08-01", "2019-08-31 23:00", tz="UTC", freq='h')

openmeteo = pd.read_pickle("Data/openmeteo.pickle")

pvgis = pd.read_pickle('Data/PVGIS.pickle')

CEDA = pd.read_pickle("CEDA_dataNL.pickle")

meteo2CEDA = {'temperature_2m' :'temperature_1_5m', 
              "relative_humidity_2m":"relative_humidity_1_5m", 
              "pressure_msl": "pressure_MSL",
              "cloud_cover":"total_cloud_amount",
              "shortwave_radiation_instant": "downward_surface_SW_flux",
              "diffuse_radiation_instant":"diffuse_surface_SW_flux",
              "direct_normal_irradiance_instant":"direct_surface_SW_flux",
              "wind_speed_10m": "wind_speed_10m",
              "wind_direction_10m": "wind_direction_10m"
              }
openmeteo = openmeteo.rename(columns=meteo2CEDA)

installation_id = "3437BD60"
prodNL = pd.read_parquet('Data/production.parquet', engine='pyarrow')
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
power = power.resample('h').sum()/4
power = power.tz_localize('UTC')
#power.index = power.index.shift(periods=2)


target_openmeteo = data_slicer(openmeteo, target_month)
eval_openmeteo = data_slicer(openmeteo, eval_month)

#TARGET
target_power = data_slicer(power, target_month)

data = pd.merge(target_power, target_openmeteo, left_index=True, right_index=True)

data = [data] # Put it in a list to work with featurisation object

#2. Featurise data & ready for training & testing 7
data = Featurisation(data)
data.data = data.cyclic_features()
data.data = data.add_shift('P')
data.data = data.cyclic_angle('wind_direction_10m')
target_dataset = data.data[0]
#EVAL

eval_power = data_slicer(power, eval_month)
data = pd.merge(eval_power, eval_openmeteo, left_index=True, right_index=True)

data = [data] # Put it in a list to work with featurisation object

#2. Featurise data & ready for training & testing 7
data = Featurisation(data)
data.data = data.cyclic_features()
data.data = data.add_shift('P')
data.data = data.cyclic_angle('wind_direction_10m')
eval_dataset = data.data[0]

scaler = MinMaxScaler()
# fit and transform the data
eval_dataset = pd.DataFrame(scaler.fit_transform(eval_dataset), columns=eval_dataset.columns)

y = eval_dataset['P']
X = eval_dataset.drop(columns = 'P')



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Regressor model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction result
y_pred_test = regressor.predict(X_test)     # predicted value of y_test
y_pred_train = regressor.predict(X_train)
plt.figure()
plt.plot(range(0,595),y_pred_train)
plt.plot(range(0,595),y_train)
# plt.show()

plt.figure()
plt.plot(range(0,149),y_pred_test)
plt.plot(range(0,149),y_test)
# plt.show()

rmse = np.sqrt(np.mean(np.square(np.subtract(y_test, y_pred_test))))
print(rmse)