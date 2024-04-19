import torch
from Models.models import source
import pandas as pd
import pickle
from hyperparameters.FeatureSelection import feature_selection
from Data.Featurisation import data_handeler
from hyperparameters.hyperparameters import hyperparameters_source
from itertools import compress
installation_id = "3437BD60"
def objective(trial, dataset, source_state_dict, scale, step, case):

    if step == 3:
        with open("hyperparameters/features.pkl", 'rb') as f:
            features = pickle.load(f)
    else:
        features = list(dataset.columns)
        features.remove('P')
    
        
    if step == 1:

    # Generate the optimizers and hyperparameters
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_loguniform("lr_source", 1e-5, 1e-1)

        n_layers = trial.suggest_int("n_layers_source", 1, 5)

        n_nodes = trial.suggest_int("n_units_source",4,502)

        dropout = trial.suggest_uniform("dropout_l", 0.1, 0.5)

        batch_size = trial.suggest_int("Batch_size_source", 4,64)

        hp = hyperparameters_source(optimizer_name, lr, n_layers, n_nodes, dropout, batch_size, trial)
    else:
        hp = hyperparameters_source()
        hp.trial = trial
        hp.load(case, 1)
        if step == 2:
            sel_features = feature_selection(trial, features)
            features = list(compress(features, sel_features))
        
        else:
            optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
            lr = trial.suggest_loguniform("lr_source", hp.lr/100, hp.lr*100)

            if hp.n_layers-2 <= 1:
                min_layers = 1
            else:
                min_layers = hp.n_layers-2
            n_layers = trial.suggest_int("n_layers_source", min_layers, hp.n_layers+2)

            n_nodes = trial.suggest_int("n_units_source",int(hp.n_nodes/2),int(hp.n_nodes*2))

            dropout = trial.suggest_uniform("dropout_l", hp.dropout/2, hp.dropout*2)

            batch_size = trial.suggest_int("Batch_size_source", int(hp.batch_size/10), int(hp.batch_size*10))
            
        



    # Make HP object

    accuracy = source(dataset, features, hp, scale)



    return accuracy




