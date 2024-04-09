from Data.Featurisation import data_handeler
from Models.models import tester
from scale import Scale
from Models.lstm import LSTM
from evaluation.evaluation import Evaluation
import matplotlib.pyplot as plt
import torch
forecast_period = 24
#### Model parameters
batch_size = 18
lr = 2.69e-3
dropout= 0.139
n_layers = 2
n_nodes = 190
optimizer_name = "Adam"

dataset_name = "nwp"
source_dataset, target_dataset, eval_dataset = data_handeler("nwp", "nwp", "nwp", False)


features= list(source_dataset.columns)
features.remove('P')
scale = Scale()
scale.load("nwp")

if "is_day" in features:
    day_index =  features.index("is_day") #BCS power also feature
    input_size = len(features)-1
else:
    day_index=None
    input_size = len(features)

target_state_dict = torch.load("Models/source")
target_model = LSTM(input_size, n_nodes, n_layers, forecast_period, dropout)
target_model.load_state_dict(target_state_dict)

y_truth, y_forecast = tester(eval_dataset, features, target_model, scale)

y_truth = y_truth.cpu().detach().flatten().numpy()
y_forecast = y_forecast.cpu().detach().flatten().numpy()

eval_obj = Evaluation(y_truth, y_forecast)
print(eval_obj.metrics())
plt.figure()

day = 1
plt.plot(y_forecast[(24*day):(24*day)+71],  label="Forecast")
plt.plot(y_truth[(24*day):(24*day)+71], label="Truth")
plt.legend()
plt.show()