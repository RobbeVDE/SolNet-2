"""
Program to simulate all models and RQ and assess them on 3 metrics: RMSE, time and R². This evaluation will be done with WFval
 1. Specify model:
    - TL(no phys)               | - target(no S, phys)) | - physical
    - TL(phys)                  | - TL(era5, no phys)   | - persistence
    - TL(no weather cov)        | - TL(era5, phys)      | - CNN-LSTM ??
    - target(no S, no phys))    | - biLSTM              | - ARIMA ??

2. Specify site:
    - 34DB760
    - 
    - 

"""
from Data.Featurisation import data_handeler
import torch
import pandas as pd
import hyperparameters.hyperparameters
from evaluation.metric_processor import metric_processor
from Models.models import target
import numpy as np
from Models import models as md
import pickle
instalid_list = ["3437BD60"]
inv_powers = [2500]
model_options_list = [(True, False, )]
rmse = pd.DataFrame()
timer = pd.DataFrame()
ctn_eval = False #Loop trough all models and sites
if ctn_eval:
    models = range(9)
    sites = range(3)
else:
    models = [int(input("Specify model:\n 0. TL(no phys)               | 4. target(no S, phys)) | 8. physical \n 1. TL(phys)                  | 5. TL(era5, no phys)   | 9. persistence \n 2. TL(no weather cov)        | 6. TL(era5, phys)      | 10. CNN-LSTM ?? \n 3. target(no S, no phys))    | 7. biLSTM              | 11. ARIMA ?? \n"))]
    sites = [int(input("Specify site: 0. 3437DB60 \n 1. ... \n 2. .... \n"))]

for i in models:
    if i <= 5:
        hp = hyperparameters.hyperparameters_target()
        hp.load(i, 3)
        match i:
            case 0:
                phys = False
                
                with open("hyperparameters/features_no_phys.pkl", 'rb') as f:
                    features = pickle.load(f)            
                hp.source_state_dict = torch.load("Models/source_nwp_no_phys")
            case 1:
                phys = True           
                with open("hyperparameters/features_phys.pkl", 'rb') as f:
                    features = pickle.load(f)
                hp.source_state_dict = torch.load("Models/source_nwp_phys")
            case 2:
                features = ["P_24h_shift", "hour_sin", "hour_cos", "month_sin", "month_cos"]
                phys = False
                hp.source_state_dict = torch.load("Models/source_no_weather")
            case 3:          
                phys = False
                source_state_dict = None
                with open("hyperparameters/features_no_phys.pkl", 'rb') as f:
                    features = pickle.load(f)
            case 4:          
                phys = True
                source_state_dict = None
                with open("hyperparameters/features_phys.pkl", 'rb') as f:
                    features = pickle.load(f)
            case 5:
                phys = False
                with open("hyperparameters/features_no_phys.pkl", 'rb') as f:
                    features = pickle.load(f)
                hp.source_state_dict = torch.load("Models/source_era5_no_phys")
            case 6:
                phys = True
                with open("hyperparameters/features_phys.pkl", 'rb') as f:
                    features = pickle.load(f)
                hp.source_state_dict = torch.load("Models/source_era5_phys")
        for j in sites:
            installation_id = instalid_list[j]
            _,_,eval_dataset = data_handeler(installation_id, "nwp", "nwp", "nwp", phys)
            accur, timer = target(eval_dataset, features, hp, WFE = True)
            metric_processor(accur, timer, i,j)

    else:
        match i:
            case 9:
                for j in sites:
                    installation_id = instalid_list[j]
                    _,_,eval_dataset = data_handeler(installation_id, "nwp", "nwp", "nwp", True)
                    accur, timer = md.persistence(eval_dataset)
                    metric_processor(accur, timer, i,j)
            case 8:
                for j in sites:
                    installation_id = instalid_list[j]
                    _,_,eval_dataset = data_handeler(installation_id, "nwp", "nwp", "nwp", True)
                    metadata = pd.read_csv("Data/installations Netherlands.csv", sep=';')
                    metadata = metadata.set_index('id')
                    metadata_id = metadata.loc[installation_id]
                    tilt = metadata_id["Tilt"]
                    peakPower = metadata_id["Watt Peak"]
                    azimuth = metadata_id["Orientation"]
                    latitude = metadata_id["Latitude"]
                    longitude = metadata_id["Longitude"]
                    accur, timer = md.physical(eval_dataset, tilt, azimuth, latitude, peakPower, inv_powers[j])
                    metric_processor(accur, timer, i,j)

        







    
        




    






    