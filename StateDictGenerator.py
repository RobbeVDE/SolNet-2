from Models.models import source
from Data.Featurisation import data_handeler
from hyperparameters.hyperparameters import hyperparameters_source
from scale import Scale
import torch

#### Model parameters

dataset_name = "nwp"
source_dataset, _, _ = data_handeler("nwp", "nwp", "nwp", True)
features = ['temperature_1_5m', 'relative_humidity_1_5m', 'diffuse_surface_SW_flux', 'direct_surface_SW_flux', 'downward_surface_SW_flux', 'P_24h_shift']

features= list(source_dataset.columns)
features.remove('P')
scale = Scale()
scale.load("nwp")

hp = hyperparameters_source()
hp.load(3)

accuracy, state_dict = source(source_dataset, features, hp, scale)

torch.save(state_dict, "Models/source")