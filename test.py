import torch
from Models.lstm import LSTM
lags = 24
forecast_period=24
hidden_size = 100
num_layers_source = 5
num_layers_target = 1
dropout = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 13
my_lstm = LSTM(input_size,hidden_size,num_layers_source,num_layers_target, forecast_period, dropout).to(device)
for param in my_lstm.source_lstm.parameters():
    param.requires_grad = False
print(my_lstm)

