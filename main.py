import pickle
import pandas as pd
from tensors.Tensorisation import Tensorisation
import torch
from evaluation.evaluation import Evaluation
from Models.training import Training
from Models.lstm import LSTM
from pvlib.pvsystem import PVSystem, FixedMount

from pvlib.modelchain import ModelChain

from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

import matplotlib.pyplot as plt
from pvlib.location import Location

from pvlib.iotools import read_tmy3

lags = 24
forecast_period=24
hidden_size = 100
num_layers = 5
dropout = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def trainer(dataset, features,  model=None, learning_rate=0.001, criterion=torch.nn.MSELoss()):
    input_size = len(features)-1

    ## Make sure that dataset is just a dataframe and not a list of dataframes
    # Get the list of features
    input_size = len(features)-1
    tensors = Tensorisation(dataset, 'P', features, lags, forecast_period)
    X_train, X_test, y_train, y_test = tensors.tensor_creation()
    
    print("Shape of data: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    
    if model==None:
        my_lstm = LSTM(input_size,hidden_size,num_layers, forecast_period, dropout).to(device)
        epochs = 80
    else:  #Avoid overfitting by reducing the amount it iterates
        my_lstm=model
        epochs = 80
    
    print(my_lstm)

    

    # Initialize the trainer
    training = Training(my_lstm, X_train, y_train, X_test, y_test, epochs, learning_rate=learning_rate, criterion=criterion)

    # Train the model and return the trained parameters and the best iteration
    state_dict_list, best_epoch = training.fit()

    return state_dict_list, best_epoch
def tester(dataset, features, model): #Here plotting possibility??
    #In evaluation data the power should be removed and can then be compared
    tensor = Tensorisation(dataset, 'P', features, lags, forecast_period)
    X, y_truth = tensor.evaluation_tensor_creation()
    model.eval()
    with torch.no_grad():
        y_forecast = model(X)
    return y_truth, y_forecast

def forecast_maker(source_data, target_data, tuning_method, features, eval_data): #, hyper_tuning, transposition,
    input_size = len(features)-1
    learning_rate=0.001
    #### SOURCE MODEL ########
 
    source_state_list, source_epoch = trainer(source_data, features)

    source_state_dict = source_state_list[source_epoch]
    
    #### TRANSFER MODEL #####
    transfer_model = LSTM(input_size,hidden_size,num_layers, forecast_period, dropout).to(device)
    transfer_model.load_state_dict(source_state_dict)
    if tuning_method == 'freeze':
        #Freeze all layers except output layer
        for param in transfer_model.lstm.parameters():
            param.requires_grad = False
        learning_rate = learning_rate/100
    elif tuning_method == "whole":
        #Change all parameters but with really low learning rated
        learning_rate = learning_rate/10e4
    else:
        raise KeyError
    target_state_list, target_epoch = trainer(target_data, features, model=transfer_model)
    target_state_dict = target_state_list[target_epoch]

    ##### TEST MODEL ######
    eval_model = LSTM(input_size,hidden_size,num_layers, forecast_period, dropout).to(device)
    eval_model.load_state_dict(target_state_dict)
    y_truth, y_forecast = tester(eval_data, features, eval_model)
    
    y_truth = y_truth.cpu().detach().flatten().numpy()
    y_forecast = y_forecast.cpu().detach().flatten().numpy()

    eval_obj = Evaluation(y_truth, y_forecast)

    print(eval_obj.metrics())

    
    return source_state_dict, target_state_dict, y_truth, y_forecast

def target_renamer(dataset, original_name):
    #Rename column with target to 'P' to simplify rest of code
    dataset = dataset.rename(columns ={original_name:'P'})
    return dataset

def CS_power(dataset, latitude, longitude, peak_power):
    PV_power = dataset['P']
    
    tz = 'UTC'

    location = Location(latitude, longitude, tz= tz)

    #1. Get clear sky irradiance for location
    times = dataset.index
    cs = location.get_clearsky(times)
    temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    # load some module and inverter specifications
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')

    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    sandia_module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']

    cec_inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    N = round(peak_power/250)
    system = PVSystem(surface_tilt=20, surface_azimuth=200,
                  module_parameters=sandia_module,
                  modules_per_string=N,
                  inverter_parameters=cec_inverter,
                  temperature_model_parameters=temperature_model_parameters)



    mc = ModelChain(system, location)
    mc.run_model(cs)
    csi_power = mc.results.dc["p_mp"]
    test = pd.merge(csi_power, PV_power, left_index = True, right_index = True)
    dataset["CS_residual"] = (test["p_mp"]-test["P"])/peak_power

    return dataset, csi_power

def eval_plotter(truth, forecast, start="2021-08-01", end="2022-08-03"):
  
    date_range = pd.date_range(start, end, freq='h')
    plt.figure()
    y1 = truth[start:end]
    y2 = forecast[start:end]
    plt.plot(date_range, y1, label='Truth')
    plt.plot(date_range, y2, label='Forecast')
    plt.xlabel("Date")
    plt.ylabel("Photovoltaic Power [kWh]")

def data_slicer(data, date_range):
    """
    Take a slice of the data which belongs to the desired date_range
    """
    data = data[data.index.isin(date_range)]

    return data