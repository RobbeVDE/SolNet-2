import torch
import optuna
from Models.models import target
import pandas as pd
from Data.Featurisation import data_handeler
from hyperparameters.hyperparameters import hyperparameters_target
from optuna.trial import TrialState
import logging
import sys
installation_id = "3437BD60"
def objective(trial, dataset, source_state_dict):

    # Generate the optimizersa and hyperparameters
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr_target = trial.suggest_loguniform("lr_target", 1e-5, 1e-1)

    n_layers_target = trial.suggest_int("n_layers_target", 1, 7)

    n_nodes_target = trial.suggest_int("n_units_target",4,1024)

    dropout = trial.suggest_uniform("dropout_target", 0.1,0.5)
    batch_size_target = trial.suggest_int("Batch_size_target", 1,64)

    min = dataset.min(axis=0).to_dict()
    max = dataset.max(axis=0).to_dict()

    
    features = list(dataset.columns)

    hp = hyperparameters_target(lr_target, n_layers_target, n_nodes_target, dropout, 
                                batch_size_target, trial, source_state_dict)

    accuracy = target(dataset, features, hp)


    return accuracy

