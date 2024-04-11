from Models.models import target
from Data.Featurisation import data_handeler
from hyperparameters.hyperparameters import hyperparameters_target
from scale import Scale
import torch

#### Model parameters
batch_size = 18
lr = 2.69e-4
dropout= 0.139
n_layers = 1
n_nodes = 40
optimizer_name = "Adam"

dataset_name = "nwp"
source_dataset, target_dataset, _ = data_handeler("nwp", "nwp", "nwp", True)

features= list(source_dataset.columns)
features.remove('P')
scale = Scale()
scale.load("nwp")

source_state_dict = torch.load("Models/source")

hp = hyperparameters_target(optimizer_name, lr, n_layers, n_nodes, dropout, batch_size, source_state_dict=source_state_dict)

accuracy, state_dict = target(target_dataset, features, hp, scale)

torch.save(state_dict, "Models/target")