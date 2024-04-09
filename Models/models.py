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

def source(dataset, features, hp, scale):
    if "is_day" in features:
        day_index =  features.index("is_day") #BCS power also feature
        input_size = len(features)-1
    else:
        day_index=None
        input_size = len(features)
    
    source_model = LSTM(input_size,hp.n_nodes,hp.n_layers, forecast_period, hp.dropout, day_index).to(device)
        
    avg_error, source_state_dict = trainer(dataset, features, model=source_model, hp=hp, scale=scale)
    if hp.trial is None:
        return avg_error, source_state_dict
    else: #Only concerned about accuracy when doing HP tuning
        return avg_error
    
def target(dataset, features, hp, scale):
    if "is_day" in features:
        day_index =  features.index("is_day") #BCS power also feature
        input_size = len(features)-1
    else:
        day_index=None
        input_size = len(features)
    transfer_model = LSTM(input_size,hp.n_nodes, hp.n_layers, forecast_period, hp.dropout, day_index).to(device)
    
    if hp.source_state_dict is not None:
        transfer_model.load_state_dict(hp.source_state_dict)
    
    avg_error, target_state_dict = trainer(dataset, features, hp, transfer_model, scale=scale)
    if hp.trial is None:
        return avg_error, target_state_dict
    else:
        return avg_error

def TL(source_data, target_data, features, eval_data, scale=None): #, hyper_tuning, transposition,
    
    if "is_day" in features:
        day_index =  features.index("is_day") #BCS power also feature
        input_size = len(features)-1
    else:
        day_index=None
        input_size = len(features)
    
    #### SOURCE MODEL ########
    source_model = LSTM(input_size,hidden_size,num_layers_source,num_layers_target, forecast_period, dropout, day_index).to(device)
    #Freeze the layers which are reserved for the target training
    
    
    source_state_list, source_epoch = trainer(source_data, features,model=source_model, scale=scale, lr=lr_source)

    source_state_dict = source_state_list[source_epoch]
    
    #### TRANSFER MODEL #####

    transfer_model = LSTM(input_size,hidden_size,num_layers_source, num_layers_target, forecast_period, dropout, day_index).to(device)
    transfer_model.load_state_dict(source_state_dict)

    for param in transfer_model.source_lstm.parameters():
        param.requires_grad = False
    
    
    target_state_list, target_epoch = trainer(target_data, features, scale=scale, model=transfer_model, lr=lr_target)
    target_state_dict = target_state_list[target_epoch]

    ##### TEST MODEL ######

    eval_model = LSTM(input_size,hidden_size,num_layers_source, num_layers_target, forecast_period, dropout, day_index).to(device)
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


def trainer(dataset, features, hp,  model=None,scale=None, criterion=torch.nn.MSELoss()):


    tensors = Tensorisation(dataset, 'P', features, lags, forecast_period, 
                            train_test_split=scale.split, domain_min=scale.min,domain_max=scale.max)
    X_train, X_test, y_train, y_test = tensors.tensor_creation()
    
    print("Shape of data: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Initialize the trainer
    training = Training(model, X_train, y_train, X_test, y_test, epochs, learning_rate=hp.lr, criterion=criterion, 
                        trial=hp.trial, optimizer_name=hp.optimizer_name, batch_size=hp.batch_size)

    # Train the model and return the trained parameters and the best iteration
    if hp.trial is None:
        avg_error, state_dict = training.fit()
    else:
        avg_error, state_dict = training.fit_cv()

    return avg_error, state_dict

def tester(dataset, features, model, scale=None): #Here plotting possibility??
    #In evaluation data the power should be removed and can then be compared
    tensor = Tensorisation(dataset, 'P', features, lags, forecast_period, domain_min=scale.min, domain_max=scale.max)
    X, y_truth = tensor.evaluation_tensor_creation()
    model.eval()
    with torch.no_grad():
        y_forecast = model(X)
    y_truth = unscale(y_truth, scale.max, scale.min)
    y_forecast = unscale(y_forecast, scale.max, scale.min)
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

    