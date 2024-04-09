import torch
from Models.models import source
import pandas as pd
import pickle
from hyperparameters.FeatureSelection import feature_selection
from Data.Featurisation import data_handeler
from hyperparameters.hyperparameters import hyperparameters_source
from itertools import compress
installation_id = "3437BD60"
def objective(trial, dataset, source_state_dict, scale, step):

    if step == 3:
        with open("hyperparameters/features_source.pkl", 'rb') as f:
            features = pickle.load(f)
    else:
        features = list(dataset.columns)
        features.remove('P')
    
    if step == 2:
        sel_features = feature_selection(trial, features)
        list(compress(features, sel_features))
        hp = hyperparameters_source()
        hp.trial = trial
        hp.load(1)
    else:

    # Generate the optimizers and hyperparameters
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_loguniform("lr_source", 1e-5, 1e-1)

        n_layers = trial.suggest_int("n_layers_source", 1, 5)

        n_nodes = trial.suggest_int("n_units_source",4,502)

        dropout = trial.suggest_uniform("dropout_l", 0.1, 0.5)

        batch_size = trial.suggest_int("Batch_size_source", 4,64)

        hp = hyperparameters_source(optimizer_name, lr, n_layers, n_nodes, dropout, batch_size, trial)

    # Make HP object

    accuracy = source(dataset, features, hp, scale)



    return accuracy




