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
def objective(trial, dataset, source_state_dict, scale, step, case):

    if step == 3:
        with open("hyperparameters/features.pkl", 'rb') as f:
            features = pickle.load(f)
    else:
        print("Only step 3 is done for target model")
        raise ValueError
    # Generate the optimizersa and hyperparameters

    lr_target = trial.suggest_loguniform("lr_target", 1e-7, 1e-1)
    dropout = trial.suggest_uniform("dropout_target", 0.1,0.5)
    batch_size_target = trial.suggest_int("Batch_size_target", 1,64)  
    hp_source = hyperparameters_source()
    hp_source.load(3)
    hp = hyperparameters_target(hp_source.optimizer_name,lr_target, hp_source.n_layers, hp_source.n_nodes, dropout, 
                                batch_size_target, trial, source_state_dict)

    accuracy = target(dataset, features, hp, scale)


    return accuracy

