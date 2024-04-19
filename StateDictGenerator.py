from Models.models import source
from Data.Featurisation import data_handeler
from hyperparameters.hyperparameters import hyperparameters_source
from scale import Scale
import torch
import pickle
#### Model parameters

dataset_name = str(input("Dataset: Enter nwp, era5 or no_weather \n"))
transfo = str(input("Use phys transfo: Enter True or False \n"))
ftr_file = "hyperparameters/features_"
case= 2 #No weather cov
if transfo in ["True", "true"]:
    transfo = True
    add_str = "phys.pkl"
    if dataset_name == "nwp":
          case=1
    elif dataset_name =="era5":
          case=6
elif transfo in ["False", "false"]:
    transfo = False
    add_str = "no_phys.pkl"
    if dataset_name == "nwp":
          case=0
    elif dataset_name =="era5":
          case=5
else:
    raise KeyError

ftr_file += add_str

source_dataset, _, _ = data_handeler(dataset_name, "nwp", "nwp", transfo)
with open(ftr_file, 'rb') as f:
            features = pickle.load(f)


scale = Scale()
scale.load(dataset_name)

hp = hyperparameters_source()
hp.load(3)
hp.gif_plotter = False
hp.bd =True
accuracy, state_dict = source(source_dataset, features, hp, scale)

torch.save(state_dict, f"Models/source_{dataset_name}_{add_str}")
