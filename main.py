import pickle
import pandas as pd

#1 Get all data in
#with open('Data/Rebase.pickle', 'rb') as f:
#    rebase = pickle.load(f)

openmeteo = pd.read_pickle("Data/openmeteo.pickle")

pvgis = pd.read_csv("Data/PVGIS demo data.csv", sep=',')
pvgis.index = pd.to_datetime(pvgis['time'], utc=True,  format='%Y%m%d:%H%M',yearfirst=True) - pd.Timedelta(minutes=10)
pvgis.drop('time', axis=1, inplace=True)

pv_power = pvgis.xs('P', axis=1)
pv_power = pv_power['2020-08-01': '2020-08-31']

data = pd.merge(pv_power, openmeteo, left_index=True, right_index=True)

data = [data] # Put it in a list to work with featurisation object

#2. Featurise data & ready for training & testing 7
from Data.Featurisation import Featurisation
data = Featurisation(data)
data.data = data.cyclic_features()
data.data = data.add_shift('P')


dataset = data.data
print(dataset)
# 3. Make tensors out of the dataset to make training faster
from tensors.Tensorisation import Tensorisation
import torch
# Get the list of features
features = list(dataset[0].columns)
lags = 24
forecast_period = 24

# Get the tensors
X_train_tot = torch.empty(0, dtype=torch.float32)
X_test_tot = torch.empty(0, dtype=torch.float32)
y_train_tot = torch.empty(0, dtype=torch.float32)
y_test_tot = torch.empty(0, dtype=torch.float32)

for i in range(len(dataset)):
    tensors = Tensorisation(dataset[i], 'P', features, lags, forecast_period)
    X_train, X_test, y_train, y_test = tensors.tensor_creation()
    X_train_tot = torch.concat([X_train_tot, X_train])
    X_test_tot = torch.concat([X_test_tot, X_test])
    y_train_tot = torch.concat([y_train_tot, y_train])
    y_test_tot = torch.concat([y_test_tot, y_test])
print(X_train_tot.shape, X_test_tot.shape, y_train_tot.shape, y_test_tot.shape)




# Import the lstm class to create an untrained LSTM
from Models.lstm import LSTM

# Set the parameters for the lstm
input_size = len(features)
hidden_size = 100
num_layers =45
dropout = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

my_lstm = LSTM(input_size,hidden_size,num_layers, forecast_period, dropout).to(device)
print(my_lstm)

# Import the training class to train the model
from Models.training import Training

# Set the training parameters
epochs = 80

# Initialize the trainer
training = Training(my_lstm, X_train_tot, y_train_tot, X_test_tot, y_test_tot, epochs, 3)

# Train the model and return the trained parameters and the best iteration
state_dict_list, best_epoch = training.fit()
 