from Models.models import target
from Data.Featurisation import data_handeler
from hyperparameters.hyperparameters import hyperparameters_target
from scale import Scale
import torch
import pickle
#### Model parameters
batch_size = 4
lr = 2.69e-5
dropout= 0.139
n_layers = 2
n_nodes = 100
optimizer_name = "Adam"

dataset_name = "nwp"
source_dataset, target_dataset, _ = data_handeler("nwp", "nwp", "nwp", True)

with open("hyperparamaters/HP_source.pkl", rb) as f:
    features = pickle.load(f)
scale = Scale()
scale.load("nwp")

source_state_dict = torch.load("Models/source")

hp = hyperparameters_target()
hp.load(3)
hp.source_state_dict = source_state_dict

accuracy, state_dict = target(target_dataset, features, hp, scale)

torch.save(state_dict, "Models/target")
