import torch
import optuna
from Models.models import target
import pandas as pd
from Data.Featurisation import data_handeler
from hyperparameters.hyperparameters import hyperparameters_target, hyperparameters_source
from hyperparameters.FeatureSelection import feature_selection 
from optuna.trial import TrialState
from itertools import compress
import logging
import pickle
import sys
installation_id = "3437BD60"
def objective(trial, dataset, source_state_dict, scale, case_n):

    match case_n:
        case 0:
            phys = False
            dataset_name = "nwp"           
        case 1:
            phys = True           
            dataset_name = "nwp"
        case 2:
            dataset_name="no_weather"
            phys = False
    
        case 5:
            phys = False
            dataset_name = "era5"
        case 6:
            phys = True
            dataset_name ="era5"
    if phys:
        phys_str = "phys.pkl"
    else:
        phys_str= "no_phys.pkl"
    ftr_str = "features/ft_"
    if case_n == 2:
        ftr_str += "no_weather"
    ftr_str += phys_str
    with open(ftr_str, 'rb') as f:
        features = pickle.load(f)


    lr_target = trial.suggest_loguniform("lr_target", 1e-8, 1e-3)
    #dropout = trial.suggest_uniform("dropout_target", 0.1,0.5)
    batch_size_target = trial.suggest_int("Batch_size_target", 1,64)
    wd = trial.suggest_loguniform("Weight_decay_target",1e-8,1e-1)
  
    hp_source = hyperparameters_source()
    hp_source.load(case_n, 3)
    hp = hyperparameters_target(hp_source.optimizer_name,lr_target, hp_source.n_layers, hp_source.n_nodes, hp_source.dropout, 
                                batch_size_target, wd, trial,source_state_dict= source_state_dict)

    accuracy = target(dataset, features, hp, scale, WFE=True)


    return accuracy

