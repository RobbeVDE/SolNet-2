from Models.models import source
from Data.Featurisation import data_handeler
from hyperparameters.hyperparameters import hyperparameters_source
from scale import Scale
import torch
import pickle
#### Model parameters

dataset_name = "nwp"
source_dataset, _, _ = data_handeler("nwp", "nwp", "nwp", False)
with open("hyperparameters/HP_source.pkl", 'rb') as f:
            features = pickle.load(f)

scale = Scale()
scale.load("nwp")

hp = hyperparameters_source()
hp.load(3)

accuracy, state_dict = source(source_dataset, features, hp, scale)

torch.save(state_dict, "Models/source")
