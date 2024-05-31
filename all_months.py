import pandas as pd
from Data.Featurisation import data_handeler
from Models.models import source, target
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
from scale import Scale
from hyperparameters.hyperparameters import hyperparameters_source, hyperparameters_target
import torch
import scienceplots
plt.style.use(['science', 'notebook'])
import warnings
import numpy as np
from tensors.Tensorisation import Tensorisation
from scale import Scale
from pvlib import location

sites = [0,1,4,5,6,7,8]
nb_years = 3
warnings. filterwarnings('ignore')
ftr_file='features/ft_no_phys.pkl' 
with open(ftr_file, 'rb') as f:
        features = pickle.load(f)
phys = False
model = 0
for site in sites:
    for start_month in range(1,13):
       
        source_data,_,eval_data = data_handeler(site, "nwp", "nwp", "nwp", phys, start_month=start_month)
        

        scale = Scale()
        scale.load(0, "nwp", phys)

        hp = hyperparameters_source()
        hp.load(model)


        accuracy_source, state_dict, timer = source(source_data, features, hp, scale)
        
        hp = hyperparameters_target()
        hp.load(model)
        hp.source_state_dict = state_dict
        accur, timer, forecasts = target(eval_data, features, hp, scale, WFE = True)
        
        df_forecasts = pd.Series(forecasts, index=eval_data.index[24:], name="P_DA")  
        df_forecasts.to_pickle(f"sensitivity_analysis/start_month/DA_no_phys_{start_month}_{site}.pkl") 

    
ftr_file='features/ft_phys.pkl' 
with open(ftr_file, 'rb') as f:
        features = pickle.load(f)

phys = True
model = 1
for site in sites:
    for start_month in range(1,13):
        source_data,_,eval_data = data_handeler(site, "nwp", "nwp", "nwp", phys, start_month=start_month)
       

        scale = Scale()
        scale.load(model, "nwp", phys)

        hp = hyperparameters_source()
        hp.load(model)


        accuracy_source, state_dict, timer = source(source_data, features, hp, scale)
        
        hp = hyperparameters_target()
        hp.load(model)
        hp.source_state_dict = state_dict
        accur, timer, forecasts = target(eval_data, features, hp, scale, WFE = True)
        
        df_forecasts = pd.Series(forecasts, index=eval_data.index[24:], name="P_DA")  
        df_forecasts.to_pickle(f"sensitivity_analysis/start_month/DA_phys_{start_month}_{site}.pkl") 

   
