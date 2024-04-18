from Data.Featurisation import data_handeler
from Models.models import target
from scale import Scale
from Models.lstm import LSTM
import pandas as pd
from hyperparameters.hyperparameters import hyperparameters_target
from evaluation.evaluation import Evaluation
import matplotlib.pyplot as plt
import torch
import pickle

forecast_period = 24
#### Model parameters
batch_size = 18
lr = 2.69e-4
dropout= 0.139
n_layers = 2
n_nodes = 100
optimizer_name = "Adam"

dataset_name = "nwp"
phys_transfo = False
source_dataset, target_dataset, eval_dataset = data_handeler(dataset_name, "nwp", "nwp", phys_transfo)

# with open("hyperparameters/features.pkl", 'rb') as f:
#     features = pickle.load(f)
features = list(source_dataset.columns)

scale = Scale()
scale.load(dataset_name)

if "is_day" in features:
    day_index =  features.index("is_day") #BCS power also feature
    input_size = len(features)-1
else:
    day_index=None
    input_size = len(features)
hp = hyperparameters_target()
hp.load(3)


source_state_dict = torch.load("Models/source_nwp")
hp.source_state_dict = source_state_dict
hp.bd = True

accuracy, state_dict, y_truth, y_forecast = target(target_dataset, features, hp, scale, WFE = True)

y_truth = y_truth.cpu().detach().flatten().numpy()
y_forecast = y_forecast.cpu().detach().flatten().numpy()



eval_obj = Evaluation(y_truth, y_forecast)
print(eval_obj.metrics())
plt.figure()

day = 1
plt.plot(y_forecast,  label="Forecast")
plt.plot(y_truth, label="Truth")
plt.legend()
plt.show()