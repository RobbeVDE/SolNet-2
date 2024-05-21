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
warnings. filterwarnings('ignore')

ftr_file='features/ft_no_phys_sa.pkl' 
with open(ftr_file, 'rb') as f:
        features = pickle.load(f)
my_index = pd.MultiIndex.from_product([range(len(features)), range(13)], names=[u'one', u'two'])
rmse_np = pd.DataFrame(index=my_index, columns=sites)
rmse_np_source = pd.DataFrame(index=range(len(features)), columns=sites)
model=0
for site in sites:
    dataset_name = "nwp"
    phys = False
    source_data, _, eval_data = data_handeler(site, 'nwp', 'nwp', 'nwp', transform=True, HP_tuning=False)



    
    scale = Scale()
    scale.load(site, dataset_name, phys)

    

    for i in range(1,len(features)):
        hp = hyperparameters_source()
        hp.load(model)
        ft = features[:i]
        accuracy_source, state_dict, timer = source(source_data, ft, hp, scale);

        hp = hyperparameters_target()
        hp.load(model)
        hp.source_state_dict = state_dict
        accur, timer, forecasts = target(eval_data, ft, hp, scale, WFE = True);

        rmse_np.loc[(i-1, slice(0,len(accur)-1)),site] = accur
        rmse_np_source.loc[i-1,site] = accuracy_source
        rmse_np.to_pickle('sensitivity_analysis/nb_features/rmse_no_physics.pkl')
        rmse_np_source.to_pickle('sensitivity_analysis/nb_features/rmse_no_physics_source.pkl')
        df_forecasts = pd.Series(forecasts, index=eval_data.index[24:], name="P_DA")  
        df_forecasts.to_pickle(f"sensitivity_analysis/nb_features/DA_no_phys.pkl") 
        print(rmse_np_source)
        print(rmse_np)



ftr_file='features/ft_phys_sa.pkl' 
with open(ftr_file, 'rb') as f:
        features = pickle.load(f)
my_index = pd.MultiIndex.from_product([range(len(features)-1), range(13)], names=[u'one', u'two'])
rmse_p = pd.DataFrame(index=my_index, columns=sites)
rmse_p_source = pd.DataFrame(index=range(len(features)), columns=sites)
model=1
for site in sites:
    dataset_name = "nwp"
    phys = True
    source_data, _, eval_data = data_handeler(site, 'nwp', 'nwp', 'nwp', transform=True, HP_tuning=False)



    
    scale = Scale()
    scale.load(site, dataset_name, phys)

    

    for i in range(2,len(features)):
        hp = hyperparameters_source()
        hp.load(model)
        ft = features[:i]
        accuracy_source, state_dict, timer = source(source_data, ft, hp, scale);

        hp = hyperparameters_target()
        hp.load(model)
        hp.source_state_dict = state_dict
        accur, timer, forecasts = target(eval_data, ft, hp, scale, WFE = True);

        rmse_p.loc[(i-2, slice(0,len(accur)-1)),site] = accur
        rmse_p_source.loc[i,site] = accuracy_source
        rmse_p.to_pickle('sensitivity_analysis/nb_features/rmse_physics.pkl')
        rmse_p_source.to_pickle('sensitivity_analysis/nb_features/rmse_physics_source.pkl')
        df_forecasts = pd.Series(forecasts, index=eval_data.index[24:], name="P_DA")  
        df_forecasts.to_pickle(f"sensitivity_analysis/nb_features/DA_phys.pkl") 
    print(rmse_p_source)
    print(rmse_p)





