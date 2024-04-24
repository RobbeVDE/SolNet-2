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
import pandas as pd
from  hyperparameters import hyperparameters
from evaluation.metric_processor import metric_processor
from Models.models import target
import numpy as np
from Models import models as md
import pickle
instalid_list = ["3437BD60"]
inv_powers = [2500]
rmse = pd.DataFrame()
timer = pd.DataFrame()
ctn_eval = False #Loop trough all models and sites
if ctn_eval:
    models = range(9)
    sites = range(4)
else:
    models = [int(input("Specify model:\n 0. TL(no phys)               | 4. target(no S, phys)) | 8. physical \n 1. TL(phys)                  | 5. TL(era5, no phys)   | 9. persistence \n 2. TL(no weather cov)        | 6. TL(era5, phys)      | 10. CNN-LSTM ?? \n 3. target(no S, no phys))    | 7. biLSTM              | 11. ARIMA ?? \n"))]
    sites = [int(input("Specify site: \n 0. NL 1       | 3. UK \n 1. NL 2        |   \n 2. Costa Rica  | \n"))]

for i in models:
    if i <= 5:
        hp = hyperparameters.hyperparameters_target()
        hp.load(i, 3)
        match i:
            case 0:
                phys = False
                dataset_name = "nwp"           
                hp.source_state_dict = torch.load(f"Models/source/nwp_{j}no_phys.pkl")
            case 1:
                phys = True           
                dataset_name = "nwp"
                hp.source_state_dict = torch.load(f"Models/source/nwp_{j}_phys.pkl")
            case 2:
                dataset_name = "no_weather"
                phys = False
                hp.source_state_dict = torch.load(f"Models/source/no_weather_{j}_no_phys.pkl")
            case 3:          
                phys = False
                dataset_name = "nwp"
                
            case 4:          
                phys = True
                dataset_name = "nwp"
            case 5:
                phys = False
                dataset_name = "era5"
                hp.source_state_dict = torch.load(f"Models/source/era5_{j}_no_phys.pkl")
            case 6:
                phys = True
                dataset_name ="era5"
                hp.source_state_dict = torch.load(f"Models/source/era5_{j}_phys.pkl")
        if phys:
            phys_str = "phys.pkl"
        else:
            phys_str= "no_phys.pkl"    
        for j in sites:
            _,_,eval_dataset = data_handeler(j, "nwp", "nwp", "nwp", phys)
            with open(f"hyperparameters/features_{dataset_name}_{phys_str}", 'rb') as f:
                    features = pickle.load(f) 
            accur, timer = target(eval_dataset, features, hp, WFE = True)
            metric_processor(accur, timer, i,j)

    else:
        match i:
            case 9:
                for j in sites:
                    _,_,eval_dataset = data_handeler(j, "nwp", "nwp", "nwp", True)
                    accur, timer = md.persistence(eval_dataset)
                    metric_processor(accur, timer, i,j)
            case 8:
                for j in sites:
                    _,_,eval_dataset = data_handeler(j, "nwp", "nwp", "nwp", True)
                    metadata = pd.read_pickle("Data/sites/metadata.pkl")
                    metadata = metadata.set_index('id')
                    metadata_id = metadata.loc[j]
                    tilt = metadata_id["Tilt"]
                    peakPower = metadata_id["Watt Peak"]
                    azimuth = metadata_id["Orientation"]
                    latitude = metadata_id["Latitude"]
                    longitude = metadata_id["Longitude"]
                    accur, timer = md.physical(eval_dataset, tilt, azimuth, latitude, peakPower, inv_powers[j])
                    metric_processor(accur, timer, i,j)

        







    
        




    






    