import pickle
import pandas as pd
from tensors.Tensorisation import Tensorisation
import torch
from evaluation.evaluation import Evaluation
from Models.training import Training
from Models.lstm import LSTM
from pvlib.pvsystem import PVSystem, FixedMount
import numpy as np
from pvlib.modelchain import ModelChain

from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

import matplotlib.pyplot as plt
from pvlib.location import Location

from pvlib.iotools import read_tmy3
epochs = 100
lags = 24
forecast_period=24
hidden_size = 400
num_layers_source = 1
num_layers_target = 5
dropout = 0.3
lr_source=0.001
lr_target = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def source(dataset, features, scale):
    if "is_day" in features:
        day_index =  features.index("is_day") #BCS power also feature
        input_size = len(features)-1
    else:
        day_index=None
        input_size = len(features)
    
    source_model = LSTM(input_size,hidden_size,num_layers_source, forecast_period, dropout, day_index).to(device)
        
    source_state_list, source_epoch = trainer(dataset, features,model=source_model, scale=scale, lr=lr_source)

    source_state_dict = source_state_list[source_epoch]

    return source_state_dict
    
def target(dataset, features, scale, source_state_dict=None):
    if "is_day" in features:
        day_index =  features.index("is_day") #BCS power also feature
        input_size = len(features)-1
    else:
        day_index=None
        input_size = len(features)
    transfer_model = LSTM(input_size,hidden_size,num_layers_source, num_layers_target, forecast_period, dropout, day_index).to(device)
    if source_state_dict is not None:
        transfer_model.load_state_dict(source_state_dict)
    

def TL(source_data, target_data, features, eval_data, trial, optimizer_name, lr_source, lr_target, n_layers_source, 
                   n_layers_target, n_nodes_source, n_nodes_target, batch_size_source, batch_size_target, dropout,scale=None): #, hyper_tuning, transposition,
    
    if "is_day" in features:
        day_index =  features.index("is_day") #BCS power also feature
        input_size = len(features)-1
    else:
        day_index=None
        input_size = len(features)
    #### SOURCE MODEL ########

    source_model = LSTM(input_size, n_layers_source, n_layers_target, n_nodes_source, n_nodes_target, forecast_period, dropout, day_index).to(device)
    #Freeze the layers which are reserved for the target training
    
    
    source_state_list, source_epoch = trainer(source_data, features,trial, optimizer_name, batch_size_source, 
                                              lr=lr_source, model=source_model, scale=scale, eval_dataset=eval_data)

    source_state_dict = source_state_list[source_epoch]
    
    #### TRANSFER MODEL #####
    transfer_model = LSTM(input_size, n_layers_source, n_layers_target, n_nodes_source, n_nodes_target, forecast_period, dropout, day_index).to(device)
    transfer_model.load_state_dict(source_state_dict)
    if n_layers_source != 0:
        for param in transfer_model.source_lstm.parameters():
            param.requires_grad = False
    
    

    target_state_list, target_epoch = trainer(target_data, features, trial, optimizer_name, batch_size_target,
                                               lr=lr_target,scale=scale, model=transfer_model, eval_dataset=eval_data)
    target_state_dict = target_state_list[target_epoch]

    ##### TEST MODEL ######
    eval_model = LSTM(input_size, n_layers_source, n_layers_target, n_nodes_source, n_nodes_target, forecast_period, dropout, day_index).to(device)
    eval_model.load_state_dict(target_state_dict)
    y_truth, y_forecast = tester(eval_data, features, eval_model, scale=scale)
    
    y_truth = y_truth.cpu().detach().flatten().numpy()
    y_forecast = y_forecast.cpu().detach().flatten().numpy()

    eval_obj = Evaluation(y_truth, y_forecast)

    
    return target_state_dict, eval_obj

def target(target_data, features, eval_data, trial, optimizer_name, lr_target,
           n_layers_target, n_nodes_target, batch_size_target, scale=None): #, hyper_tuning, transposition,
    if "is_day" in features:
        day_index =  features.index("is_day") #BCS power also feature
        input_size = len(features)-1
    else:
        input_size = len(features)
        
    #### TRANSFER MODEL #####

    transfer_model = LSTM(input_size, 0, n_layers_target, 0, n_nodes_target, forecast_period, day_index).to(device)

    
    target_state_list, target_epoch = trainer(target_data, features, trial, optimizer_name, batch_size_target, lr=lr_target,scale=scale, model=transfer_model, eval_dataset=eval_data)
    target_state_dict = target_state_list[target_epoch]

    ##### TEST MODEL ######

    eval_model = LSTM(input_size, 0, n_layers_target, 0, n_nodes_target, forecast_period, dropout, day_index).to(device)
    eval_model.load_state_dict(target_state_dict)
    y_truth, y_forecast = tester(eval_data, features, eval_model, scale=scale)
    
    y_truth = y_truth.cpu().detach().flatten().numpy()
    y_forecast = y_forecast.cpu().detach().flatten().numpy()

    eval_obj = Evaluation(y_truth, y_forecast)

    
    return target_state_dict, eval_obj

def persistence(dataset):

    y_forecast = dataset['P_24h_shift']
    y_truth = dataset['P']
    eval_obj = Evaluation(y_truth, y_forecast)

    return eval_obj


def trainer(dataset, features, trial, optimizer_name, batch_size=32, model=None,scale=None, lr=0.001, criterion=torch.nn.MSELoss(), eval_dataset=None):


    tensors = Tensorisation(dataset, 'P', features, lags, forecast_period, domain_min=scale[0], domain_max=scale[1])
    X_train, X_test, y_train, y_test = tensors.tensor_creation()
    print("Shape of data: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    if eval_dataset is not None:
        tensors =  Tensorisation(eval_dataset, 'P', features, lags, forecast_period, domain_min=scale[0], domain_max=scale[1])
        X_eval, y_eval = tensors.evaluation_tensor_creation()

    # Initialize the trainer
    training = Training(model, X_train, y_train, X_test, y_test, X_eval, y_eval, epochs, optimizer_name, trial, batch_size=batch_size, 
                        learning_rate=lr, criterion=criterion)
    # Train the model and return the trained parameters and the best iteration
    state_dict_list, best_epoch = training.fit()

    return state_dict_list, best_epoch

def tester(dataset, features, model, scale=None): #Here plotting possibility??
    #In evaluation data the power should be removed and can then be compared
    tensor = Tensorisation(dataset, 'P', features, lags, forecast_period, domain_min=scale[0], domain_max=scale[1])
    X, y_truth = tensor.evaluation_tensor_creation()
    model.eval()
    with torch.no_grad():
        y_forecast = model(X)
    y_truth = unscale(y_truth, scale[1], scale[0])
    y_forecast = unscale(y_forecast, scale[1], scale[0])
    return y_truth, y_forecast




def CS_power(dataset, latitude, longitude, peak_power):
    PV_power = dataset['P']
    
    tz = 'UTC'

    location = Location(latitude, longitude, tz= tz)

    #1. Get clear sky irradiance for location
    times = dataset.index
    cs = location.get_clearsky(times)
    temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    # load some module and inverter specifications (this is random for now)
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



def unscale(y, max, min):
    max= max['P']
    min = min['P']
    y = y*(max-min) + min

    return y

    
