import torch
from Models.models import source
import pandas as pd
from Data.Featurisation import data_handeler
from hyperparameters.hyperparameters import hyperparameters_source
installation_id = "3437BD60"
def objective(trial, dataset, source_state_dict):

    # Generate the optimizers and hyperparameters
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("lr_source", 1e-5, 1e-1)

    n_layers = trial.suggest_int("n_layers_source", 1, 5)

    n_nodes = trial.suggest_int("n_units_source",4,502)

    dropout = trial.suggest_uniform("dropout_l", 0.1, 0.5)

    batch_size = trial.suggest_int("Batch_size_source", 4,64)
    # Get data

    min = dataset.min(axis=0).to_dict()
    max = dataset.max(axis=0).to_dict()

    
    features = list(dataset.columns)

    # Make HP object

    hp = hyperparameters_source(optimizer_name, lr, n_layers, n_nodes, dropout, batch_size, trial)
    accuracy = source(dataset, features, hp)



    return accuracy




