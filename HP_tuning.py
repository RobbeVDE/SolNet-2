from optuna.trial import TrialState
from optuna.samplers import TPESampler
from Data.Featurisation import data_handeler
import torch
import logging
import sys
import pickle
import optuna
from functools import partial

tuning_model = "source"  # Unique identifier of the study.
dataset_name = "era5"
transfo = True
TL = True
source_state_dict = None
source_data, target_data, _ = data_handeler(dataset_name, dataset_name, dataset_name, transfo)
if tuning_model == "source":
    from hyperparameters.OptunaSource import objective
    dataset = source_data
elif tuning_model == "target":
    from hyperparameters.OptunaTa import objective
    dataset = target_data
    if TL:
        source_state_dict = torch.load("Models/source")

objective = partial(objective,  dataset = dataset, source_state_dict = source_state_dict)

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

storage_name = f"sqlite:///HP_{tuning_model}.db"
study_name = f"{dataset_name} | transfo: {transfo} | TL: {TL}"
try:
    restored_sampler = pickle.load(open(f"hyperparameters/samplers/sampler_{tuning_model}_{dataset_name}_{transfo}.pkl", "rb")) 
except: #If there is no sampler present, make a new one
    sampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.

study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", 
                            load_if_exists=True, sampler=sampler)
try:
    study.optimize(objective, n_trials=100)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
except KeyboardInterrupt: #If optimization process gets interrupted the sampler is saved for next time 
    with open(f"hyperparameters/samplers/sampler_{tuning_model}_{dataset_name}_{transfo}_{TL}.pkl", "wb") as fout: 
        pickle.dump(study.sampler, fout)

