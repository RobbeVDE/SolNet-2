import pandas as pd
import numpy as np
from tensors.Tensorisation import NWP_tensorisation
from Models.lstm import LSTM
lags = 24
forecast_period = 24

ceda = pd.read_pickle("CEDA_dataNL.pickle")
openmeteo = pd.read_pickle("Data/openmeteo.pickle")
pvgis = pd.read_pickle('Data/PVGIS.pickle')

prod_NL = pd.read_pickle("Data/NL_power.pickle")
meteo2CEDA = {'temperature_2m' :'temperature_1_5m', 
              "relative_humidity_2m":"relative_humidity_1_5m", 
              "pressure_msl": "pressure_MSL",
              "cloud_cover":"total_cloud_amount",
              "shortwave_radiation": "downward_surface_SW_flux",
              "diffuse_radiation":"diffuse_surface_SW_flux",
              "direct_normal_irradiance":"direct_surface_SW_flux",
              "wind_speed_10m": "wind_speed_10m",
              "wind_direction_10m": "wind_direction_10m"
              }
openmeteo = openmeteo.rename(columns=meteo2CEDA)

features = ['downward_surface_SW_flux', 'direct_surface_SW_flux', 'diffuse_surface_SW_flux']

openmeteo = openmeteo.reindex(ceda.index)

tensors = NWP_tensorisation(ceda, openmeteo, features, lags, forecast_period)

X_train, X_test, Y_train, Y_test = tensors.tensor_creation()

input_size = len(features)
hidden_size = 50
n_layers = 1
output_size = forecast_period
dropout = 0.1

model1 = LSTM(input_size, hidden_size, n_layers, output_size, dropout)

