import optuna
import os

tuning_model = str(input("Tuning Model: Enter source or target \n"))  # Unique identifier of the study.
dataset_name = str(input("Dataset: Enter nwp or era5 \n"))
transfo = str(input("Use phys transfo: Enter True or False \n"))

if transfo in ["True", "true"]:
    transfo = True
elif transfo in ["False", "false"]:
    transfo = False
else:
    raise KeyError

TL = str(input("TL case: Enter True or False \n"))
if TL in ["True", "true"]:
    TL = True
elif TL in ["False", "false"]:
    TL = False
else:
    raise KeyError
step = int(input("Select step in optimization: \n 1: Initial HP \n 2: Feature Selection \n 3: Complete HP \n"))

#Remove study object
storage_name = f"sqlite:///HP_{tuning_model}.db"
study_name = f"{dataset_name} | transfo: {transfo} | TL: {TL} | Step: {step}"
    
optuna.delete_study(study_name=study_name, storage=storage_name)


#Remove sampler
filepath = f"hyperparameters/samplers/sampler_{tuning_model}_{dataset_name}_{transfo}_{TL}_{step}.pkl"
if os.path.exists(filepath):
  os.remove(f"hyperparameters/samplers/sampler_{tuning_model}_{dataset_name}_{transfo}_{TL}_{step}.pkl")
