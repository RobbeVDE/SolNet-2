from Models.models import source
from Data.Featurisation import data_handeler
from hyperparameters.hyperparameters import hyperparameters_source
from scale import Scale
import torch

#### Model parameters
batch_size = 4
lr = 1.75e-3
dropout= 0.11
n_layers = 2
n_nodes = 190
optimizer_name = "RMSprop"

dataset_name = "nwp"
source_dataset, _, _ = data_handeler("nwp", "nwp", "nwp", False)

features= list(source_dataset.columns)
features.remove('P')
scale = Scale()
scale.load("nwp")

hp = hyperparameters_source(optimizer_name, lr, n_layers, n_nodes, dropout, batch_size)

accuracy, state_dict = source(source_dataset, features, hp, scale)

torch.save(state_dict, "Models/source")