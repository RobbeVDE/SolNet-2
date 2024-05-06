import torch
from Models.models import source
import pandas as pd
import pickle
from hyperparameters.FeatureSelection import feature_selection
from Data.Featurisation import data_handeler
from hyperparameters.hyperparameters import hyperparameters_source
from itertools import compress
installation_id = "3437BD60"
def objective(trial, dataset, source_state_dict, scale, case_n):

    phys_str=""
    match case_n:
        case 0:
            phys = False
            dataset_name = "nwp"           
        case 1:
            phys = True           
            dataset_name = "nwp"
        case 2:
            dataset_name = "no_weather"
            phys = False
            phys_str +="no_weather_"
    
        case 5:
            phys = False
            dataset_name = "era5"
        case 6:
            phys = True
            dataset_name ="era5"
    if phys:
        phys_str += "phys.pkl"
    else:
        phys_str += "no_phys.pkl"    
    with open(f"features/ft_{phys_str}", 'rb') as f:
        features = pickle.load(f)
 
# Generate the optimizers and hyperparameters
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("lr_source", 1e-6, 1e-1)

    n_layers = trial.suggest_int("n_layers_source", 1, 3)

    n_nodes = trial.suggest_int("n_units_source",4,800)

    dropout = trial.suggest_uniform("dropout_l", 0.1, 0.5)

    batch_size = trial.suggest_int("Batch_size_source", 4,128)

    wd = trial.suggest_loguniform("Weight_decay_source",1e-8,1e-1)

    hp = hyperparameters_source(optimizer_name, lr, n_layers, n_nodes, dropout, batch_size, wd, trial)
    
        



    # Make HP object

    accuracy = source(dataset, features, hp, scale)



    return accuracy




