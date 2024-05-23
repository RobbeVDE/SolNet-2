import pandas as pd
from Data.Featurisation import data_handeler
from Models.models import source, target
import pickle
import os
from scale import Scale
from hyperparameters.hyperparameters import hyperparameters_source, hyperparameters_target
import torch
import warnings
import numpy as np

sites = range(4)
max_nb_years = 10
warnings. filterwarnings('ignore')
ftr_file='features/ft_no_phys.pkl' 
with open(ftr_file, 'rb') as f:
        features = pickle.load(f)
my_index = pd.MultiIndex.from_product([range(max_nb_years), range(13)], names=[u'one', u'two'])
rmse_np = pd.DataFrame(index=my_index, columns=sites)
rmse_np_source = pd.DataFrame(index=range(max_nb_years), columns=sites)
phys = False
model = 5
for site in sites:
    for i in range(1,max_nb_years):
        source_data,_,eval_data = data_handeler(site, "era5", "nwp", "nwp", phys, nb_source_years=i)
       

        scale = Scale()
        scale.load(0, "era5", phys)

        hp = hyperparameters_source()
        hp.load(model)


        accuracy_source, state_dict, timer = source(source_data, features, hp, scale);
        
        hp = hyperparameters_target()
        hp.load(model)
        hp.source_state_dict = state_dict
        accur, timer, forecasts = target(eval_data, features, hp, scale, WFE = True);
        rmse_np.loc[(i-1, slice(0,len(accur)-1)),site] = accur
        rmse_np_source.loc[i-1,site] = accuracy_source
        rmse_np.to_pickle('sensitivity_analysis/nb_years/rmse_no_physics.pkl')
        rmse_np_source.to_pickle('sensitivity_analysis/nb_years/rmse_no_physics_source.pkl')
        df_forecasts = pd.Series(forecasts, index=eval_data.index[24:], name="P_DA")  
        df_forecasts.to_pickle(f"sensitivity_analysis/nb_years/DA_no_phys_{i}_{site}.pkl") 
        print(rmse_np_source)
        print(rmse_np)
    
ftr_file='features/ft_phys.pkl' 
with open(ftr_file, 'rb') as f:
        features = pickle.load(f)
my_index = pd.MultiIndex.from_product([range(max_nb_years), range(13)], names=[u'one', u'two'])
rmse_np = pd.DataFrame(index=my_index, columns=sites)
rmse_np_source = pd.DataFrame(index=range(max_nb_years), columns=sites)
phys = True
model = 6
for site in sites:
    for i in range(1,max_nb_years):
        source_data,_,eval_data = data_handeler(site, "era5", "nwp", "nwp", phys, nb_source_years=i)
       

        scale = Scale()
        scale.load(0, "era5", phys)

        hp = hyperparameters_source()
        hp.load(model)


        accuracy_source, state_dict, timer = source(source_data, features, hp, scale);
        
        hp = hyperparameters_target()
        hp.load(model)
        hp.source_state_dict = state_dict
        accur, timer, forecasts = target(eval_data, features, hp, scale, WFE = True);
        rmse_np.loc[(i-1, slice(0,len(accur)-1)),site] = accur
        rmse_np_source.loc[i-1,site] = accuracy_source
        rmse_np.to_pickle('sensitivity_analysis/nb_years/rmse_physics.pkl')
        rmse_np_source.to_pickle('sensitivity_analysis/nb_years/rmse_physics_source.pkl')
        df_forecasts = pd.Series(forecasts, index=eval_data.index[24:], name="P_DA")  
        df_forecasts.to_pickle(f"sensitivity_analysis/nb_years/DA_phys_{i}_{site}.pkl") 
        print(rmse_np_source)
        print(rmse_np)
   