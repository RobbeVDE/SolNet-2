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
source_dataset, target_dataset, _ = data_handeler(dataset_name, "nwp", "nwp", True)

with open("hyperparameters/features.pkl", 'rb') as f:
    features = pickle.load(f)
scale = Scale()
scale.load(dataset_name)

source_state_dict = torch.load(f"Models/source_{dataset_name}")

hp = hyperparameters_target()
hp.load(3)
#hp.source_state_dict = source_state_dict

accuracy, state_dict,_,_ = target(target_dataset, features, hp, scale, True)

torch.save(state_dict, f"Models/target")
