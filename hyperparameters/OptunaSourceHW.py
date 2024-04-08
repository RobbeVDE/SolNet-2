import torch
from Models.models import source
import pandas as pd
from Data.Featurisation import data_handeler

installation_id = "3437BD60"
def objective(trial):

    # Generate the optimizersa and hyperparameters
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("lr_source", 1e-5, 1e-1)

    n_layers = trial.suggest_int("n_layers_source", 1, 7)

    n_nodes = trial.suggest_int("n_units_source",4,1024)

    dropout = trial.suggest_uniform("dropout_l", 0.1, 0.5)

    batch_size = trial.suggest_int("Batch_size_source", 4,64)
    # Get data

    source_dataset, target_dataset, eval_dataset = data_handeler("openmeteo", "ceda", "ceda", transform=True, month_source=False)
    min = source_dataset.min(axis=0).to_dict()
    max = source_dataset.max(axis=0).to_dict()

    
    features = list(source_dataset.columns)
    features = ['temperature_1_5m', 'relative_humidity_1_5m', 'diffuse_surface_SW_flux', 'direct_surface_SW_flux', 'downward_surface_SW_flux', 'PoA', 'P_24h_shift', "is_day"]

    
    accuracy = source(source_dataset, features, trial, optimizer_name, lr, n_layers, n_nodes, batch_size, dropout)



    return accuracy




