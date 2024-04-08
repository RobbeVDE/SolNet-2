from Models.models import source
from Data.Featurisation import data_handeler
import torch

#### Model parameters
batch_size = 5
lr = 0.00138
dropout= 0.364
n_layers = 1
n_nodes = 316
optimizer_name = "Adam"


source_dataset, _, _ = data_handeler("ceda", "ceda", "ceda")

features = features = ['temperature_1_5m', 'relative_humidity_1_5m', 'diffuse_surface_SW_flux', 'direct_surface_SW_flux', 'downward_surface_SW_flux', 'PoA', 'P_24h_shift', "is_day"]

accuracy, state_dict = source(source_dataset, features, optimizer_name, lr, n_layers, n_nodes, batch_size, dropout)

torch.save(state_dict, "Models/source")