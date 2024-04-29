"""
Program to simulate all models and RQ and assess them on 3 metrics: RMSE, time and R². This evaluation will be done with WFval
 1. Specify model:
    - TL(no phys)               | - target(no S, phys)) | - physical
    - TL(phys)                  | - TL(era5, no phys)   | - persistence
    - TL(no weather cov)        | - TL(era5, phys)      | - CNN-LSTM ??
    - target(no S, no phys))    | - biLSTM              | - ARIMA ??

2. Specify site:
    - NL 1          | UK
    - NL 2          |
    - Costa Rica    |

"""
from Data.Featurisation import data_handeler
import torch
from scale import Scale
import pandas as pd
from  hyperparameters import hyperparameters
from evaluation.metric_processor import metric_processor_target
from Models.models import target
import numpy as np
import os
from Models import models as md
import pickle
rmse = pd.DataFrame()
timer = pd.DataFrame()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ctn_eval = False #Loop trough all models and sites
if ctn_eval:
    models = range(9)
    sites = range(4)
else:
    models = [int(input("Specify model:\n 0. TL(no phys)               | 4. target(no S, phys)) | 8. physical \n 1. TL(phys)                  | 5. TL(era5, no phys)   | 9. persistence \n 2. TL(no weather cov)        | 6. TL(era5, phys)      | 10. CNN-LSTM ?? \n 3. target(no S, no phys))    | 7. biLSTM              | 11. ARIMA ?? \n"))]
    sites = [int(input("Specify site: \n 0. NL 1       | 3. UK \n 1. NL 2        |   \n 2. Costa Rica  | \n"))]
for i in models:
    for j in sites:
        if i <= 5:
            hp = hyperparameters.hyperparameters_target()
            try:
                hp.load(i, 3)
            except:
                hp.load(0, 3)
            match i:
                case 0:
                    phys = False
                    dataset_name = "nwp"           
                    hp.source_state_dict = torch.load(f"Models/source/nwp_{j}_no_phys.pkl", map_location=device)
                case 1:
                    phys = True           
                    dataset_name = "nwp"
                    hp.source_state_dict = torch.load(f"Models/source/nwp_{j}_phys.pkl", map_location=device)
                case 2:
                    dataset_name = "no_weather"
                    phys = False
                    hp.source_state_dict = torch.load(f"Models/source/no_weather_{j}_no_phys.pkl", map_location=device)
                case 3:          
                    phys = False
                    dataset_name = "nwp"
                    
                case 4:          
                    phys = True
                    dataset_name = "nwp"
                case 5:
                    phys = False
                    dataset_name = "era5"
                    hp.source_state_dict = torch.load(f"Models/source/era5_{j}_no_phys.pkl", map_location=device)
                case 6:
                    phys = True
                    dataset_name ="era5"
                    hp.source_state_dict = torch.load(f"Models/source/era5_{j}_phys.pkl", map_location=device)
            if phys:
                phys_str = "phys.pkl"
            else:
                phys_str= "no_phys.pkl"    
            _,_,eval_dataset = data_handeler(j, "nwp", "nwp", "nwp", phys)
            ftr_string ="hyperparameters/features_"

            if i == 2:
                ftr_string+= "no_weather_"
            ftr_string += phys_str
            scale = Scale()
            scale.load(j, dataset_name)
            if os.path.isfile(ftr_string):
                with open(ftr_string, 'rb') as f:
                        features = pickle.load(f)
            else:
                features = list(eval_dataset.columns)
                features.remove('P')
            accur, timer = target(eval_dataset, features, hp, scale, WFE = True)
            metric_processor_target(accur, timer, i,j)

    else:
        match i:
            case 9:
                for j in sites:
                    _,_,eval_dataset = data_handeler(j, "nwp", "nwp", "nwp", False)
                    accur, timer = md.persistence(eval_dataset)
                    metric_processor_target(accur, timer, i,j)
            case 8:
                for j in sites:
                    _,_,eval_dataset = data_handeler(j, "nwp", "nwp", "nwp", True)
                    metadata = pd.read_pickle("Data/sites/metadata.pkl")
                    metadata_id = metadata.loc[j]
                    tilt = metadata_id["Tilt"]
                    peakPower = metadata_id["Installed Power"]
                    azimuth = metadata_id["Azimuth"]
                    lat = metadata_id["Latitude"]
                    lon = metadata_id["Longitude"]
                    inv_power = metadata_id["Inverter Power"]
                    accur, timer = md.physical(eval_dataset, tilt, azimuth, peakPower, inv_power, latitude=lat, longitude=lon)
                    metric_processor_target(accur, timer, i,j)

        







    
        




    






    