import pandas as pd
import numpy as np
from tensors.Tensorisation import NWP_tensorisation
from Models.lstm import LSTM
from Models.training import Training
from evaluation.evaluation import Evaluation
import torch
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

eval1 = Evaluation(openmeteo["downward_surface_SW_flux"], ceda["downward_surface_SW_flux"])
print(eval1.metrics())

y = openmeteo["downward_surface_SW_flux"]
min = min(y.iloc[:int(0.8*len(y.index))])
max = max(y.iloc[:int(0.8*len(y.index))])
X_train, X_test, Y_train, Y_test = tensors.tensor_creation()

input_size = len(features)
hidden_size = 200
n_layers = 2
output_size = forecast_period
dropout = 0.1
epochs = 100
opt_name = "Adam"
batch_size = 40
lr = 0.0005

model = LSTM(input_size, hidden_size, n_layers, output_size, dropout)
training = Training(model, X_train,  Y_train[:,:,0], X_test, Y_test[:,:,0], epochs, opt_name, batch_size=batch_size, learning_rate=lr)

error, state_dict = training.fit()

model.load_state_dict(state_dict)
model.eval()
with torch.inference_mode():
    y_predict = model(X_test)
y_truth = Y_test[:,:,0]
y_truth = y_truth.cpu().detach().flatten().numpy()
y_predict = y_predict.cpu().detach().flatten().numpy()

y_truth = min + y_truth * (max-min)
y_predict = min + y_predict * (max-min)
eval2 = Evaluation(y_truth, y_predict)
print(eval2.metrics())
