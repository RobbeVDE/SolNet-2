import pickle
import pandas as pd
from tensors.Tensorisation import Tensorisation
import torch
import evaluation.evaluation as evl
from evaluation.timer import Timer
from Models.training import Training
from Models.lstm import LSTM
from pvlib.pvsystem import PVSystem, FixedMount
import numpy as np
import optuna
from pvlib.modelchain import ModelChain
from pvlib import temperature, irradiance, location, pvsystem
from sklearn.metrics import r2_score, mean_squared_error
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

import matplotlib.pyplot as plt
from pvlib.location import Location

from pvlib.iotools import read_tmy3
epochs = 100
lags = 24
forecast_period=24


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def source(dataset, features, hp, scale):
    if "is_day" in features:
        day_index =  features.index("is_day") #BCS power also feature
        input_size = len(features)-1
    else:
        day_index=None
        input_size = len(features)
    
    source_model = LSTM(input_size,hp.n_nodes,hp.n_layers, forecast_period, hp.dropout, hp.bd, day_index).to(device)
        
    avg_error, source_state_dict, timer = trainer(dataset, features, model=source_model, hp=hp, scale=scale)
    if hp.trial is None:
        return avg_error, source_state_dict, timer
    else: #Only concerned about accuracy when doing HP tuning
        return avg_error
    
def target(dataset, features, hp, scale, WFE):
    if "is_day" in features:
        day_index =  features.index("is_day") #BCS power also feature
        input_size = len(features)-1
    else:
        day_index=None
        input_size = len(features)
    transfer_model = LSTM(input_size,hp.n_nodes, hp.n_layers, forecast_period, hp.dropout, hp.bd, day_index).to(device)
    
    if hp.source_state_dict is not None:
        transfer_model.load_state_dict(hp.source_state_dict)
    
    if WFE:
        avg_error, times = WF_trainer(dataset, features, hp, transfer_model, scale=scale) 
    else:
        avg_error, times = trainer(dataset, features, hp, transfer_model, scale=scale)
    if hp.trial is None:
        return avg_error, times
    else:
        return avg_error


def persistence(dataset):
    infer_timer = Timer()
    y_forecast = dataset['P_24h_shift']
    infer_timer.stop()
    y_truth = dataset['P']
    powers = pd.concat([y_truth, y_forecast], axis=1)
    powers = powers.iloc[lags:] #Lag values cannot be taken into account
    powers['train_month'] = [i//(30*24) for i in range(len(powers.index))] #Our months in training process consists of 30 days 
    error = powers.groupby(powers.train_month).apply(r2_rmse).reset_index()
    error = error['rmse'].to_list()
    print(error)
    times = {'Inference Time': [infer_timer.elapsed_time()/13]*13} #Just assume it takes same amount of time for each month which makes sense

    return error, times


def physical(dataset, tilt, azimuth, peakPower, peakInvPower, temp_coeff=-0.004, loss_inv=0.96, latitude=None, longitude=None):
    
    try:
        poa = dataset['PoA']
    except:
        azimuth = azimuth+180 #PVGIS works in [-180,180] and pvlib in [0,360]

        site = location.Location(latitude, longitude,  tz='UTC')
        
        times = dataset.index
        solar_position = site.get_solarposition(times=times)
        ghi = dataset["downward_surface_SW_flux"]
        dhi = dataset["diffuse_surface_SW_flux"]
        dni = dataset["direct_surface_SW_flux"]
        dni_extra = irradiance.get_extra_radiation(times)
        POA_irrad = irradiance.get_total_irradiance(surface_tilt=tilt, surface_azimuth=azimuth, 
                                                    dni=dni, ghi=ghi, dhi=dhi, solar_zenith=solar_position['apparent_zenith'],
                                                    dni_extra=dni_extra, model='perez',solar_azimuth=solar_position['azimuth'])
        poa = POA_irrad['poa_global'].fillna(0)

    infer_timer = Timer()
    temp = dataset['temperature_1_5m'] -275.13
    wind_speed = dataset['wind_speed_10m']
    wind_height = 10

    temp_cell = temperature.fuentes(poa, temp, wind_speed, 49, wind_height=wind_height,
                                          surface_tilt=tilt)
    
    inv_params = {'pdc0': peakInvPower, 'eta_inv_nom': loss_inv}
    module_params = {'pdc0': peakPower, 'gamma_pdc': temp_coeff}
    mount = pvsystem.FixedMount(surface_tilt=tilt, surface_azimuth=azimuth)
    array = pvsystem.Array(mount=mount, module_parameters = module_params)
    pvsys = pvsystem.PVSystem(arrays=[array], inverter_parameters=inv_params)
    dc_power = pvsys.pvwatts_dc(poa, temp_cell)
    forecast = pvsys.get_ac('pvwatts', dc_power)
  
    infer_timer.stop()
    actual = dataset[['P']]
    powers = pd.concat([actual, forecast], axis=1)
    powers['train_month'] = [i//(30*24) for i in range(len(powers.index))] #Our months in training process consists of 30 days 
    error = powers.groupby(powers.train_month).apply(r2_rmse).reset_index()
    error = error['rmse'].to_list()
    times = {'Inference Time': [infer_timer.elapsed_time()/13]*13} #Just assume it takes same amount of time for each month which makes sense




    return error, times

def r2_rmse(g):
    rmse = np.sqrt(mean_squared_error(g.iloc[:,0], g.iloc[:,1]))
    return pd.Series(dict(rmse = rmse))

def trainer(dataset, features, hp,  model=None,scale=None, criterion=torch.nn.MSELoss()):


    tensors = Tensorisation(dataset, 'P', features, lags, forecast_period, domain_min=scale.min,domain_max=scale.max)
    X_train, X_test, y_train, y_test = tensors.tensor_creation()
    
    # X_train = torch.flip(X_test, [0])
    # X_test = torch.flip(X_train, [0])
    # y_train = torch.flip(y_test, [0])
    # y_test = torch.flip(y_train, [0])
    print("Shape of data: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    if hp.gif_plotter:
        infer_day = 38 #Just a random day
    else:
        infer_day =None

    # Initialize the trainer
    training = Training(model, X_train, y_train, X_test, y_test, epochs, learning_rate=hp.lr, criterion=criterion, 
                        trial=hp.trial, optimizer_name=hp.optimizer_name, batch_size=hp.batch_size, infer_day=infer_day)

    # Train the model and return the trained parameters and the best iteration
    if hp.trial is None:
        timer = Timer()
        avg_error, state_dict = training.fit()
        timer.stop()
    else:
        timer = Timer()
        avg_error, state_dict = training.fit_cv()
        timer.stop()
        

    return avg_error, state_dict, timer.elapsed_time()

def WF_trainer(dataset, features, hp,  model=None,scale=None, criterion=torch.nn.MSELoss()):


    tensors = Tensorisation(dataset, 'P', features, lags, forecast_period,  domain_min=scale.min,domain_max=scale.max)
    X_train, X_test, y_train, y_test = tensors.tensor_creation(WFE=True)
    
    print("Shape of data: ", X_train[0].shape, X_test[0].shape, y_train[0].shape, y_test[0].shape)

    y_forecast = torch.zeros(len(X_test),30,24)
    # Initialize the trainer
    mse = []
    inf_times = []
    training_times = []
    try:
        cs_power = dataset["CS_power"]
        cs_power = cs_power.to_numpy()
        cs_power = cs_power[lags:] #Cannot include first data bcs these are lags
        phys = True
    except:
        phys=False
    for i in range(len(X_test)):
        model.eval()
        
        with torch.inference_mode():
            inf_timer = Timer()
            y_interm = model(X_test[i])
            inf_timer.stop()
        y_interm = y_interm.squeeze()
        # y_forecast[i,:,:] = y_interm
        inf_times.append(inf_timer.elapsed_time())

        y_f_mse = y_interm.cpu().detach().flatten().numpy()
        y_t_mse = y_test[i].cpu().detach().flatten().numpy()

        # Physical post-processing power should be between 0 and CS power
        if phys:
            y_f_mse[y_f_mse<0] = 0
            y_f_mse[y_f_mse>1] = 1

            #Unscale it again to have values in Watt
            if ((i+1)*24*30) < len(cs_power):
                y_max = cs_power[i*24*30:(i+1)*24*30]
            else:
                y_max = cs_power[i*24*30:]
        else:
            y_max = 1
        y_f_mse = unscale(y_f_mse, y_max, scale.max, scale.min)
        y_t_mse = unscale(y_t_mse, y_max, scale.max, scale.min)


        mse.append(np.mean(np.square(y_f_mse-y_t_mse)))
        print(f"Currently in month {i}. MSE for this month is: {mse[-1]}")
        if hp.trial is not None:
            hp.trial.report(mse[-1], i)
            if hp.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        if i != len(X_test)-1:
            train_timer = Timer()
            training = Training(model, X_train[i], y_train[i], None, None, epochs, learning_rate=hp.lr, criterion=criterion, 
                            trial=hp.trial, optimizer_name=hp.optimizer_name, batch_size=hp.batch_size)
            avg_error, state_dict = training.fit()
            train_timer.stop()
            model.load_state_dict(state_dict)
            training_times.append(train_timer.elapsed_time())
        else:
            training_times.append(np.nan)

    
    if hp.trial is not None: #If HP report average mse
        error = np.mean(mse)
    else:
        error  = np.sqrt(mse)

    times = {'Inference Time':inf_times, 'Training Time': training_times}

    return error, times

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






def eval_plotter(truth, forecast, start="2021-08-01", end="2022-08-03"):
  
    date_range = pd.date_range(start, end, freq='h')
    plt.figure()
    y1 = truth[start:end]
    y2 = forecast[start:end]
    plt.plot(date_range, y1, label='Truth')
    plt.plot(date_range, y2, label='Forecast')
    plt.xlabel("Date")
    plt.ylabel("Photovoltaic Power [kWh]")



def unscale(y, cs_power, max, min):
    max= max['P']
    min = min['P']
    y = y*(max-min) + min
    y = y*cs_power


    return y


    