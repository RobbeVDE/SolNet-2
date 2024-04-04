import torch
import optuna
from Models.models import target
import pandas as pd
from Data.Featurisation import data_handeler
from optuna.trial import TrialState
import logging
import sys
installation_id = "3437BD60"
def objective(trial):

    # Generate the optimizersa and hyperparameters
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr_target = trial.suggest_loguniform("lr_target", 1e-5, 1e-1)

    n_layers_target = trial.suggest_int("n_layers_target", 1, 7)

    n_nodes_target = trial.suggest_int("n_units_target",4,1024)


    batch_size_target = trial.suggest_int("Batch_size_target", 1,64)
    # Get data

    source_dataset, target_dataset, eval_dataset = data_handeler("ceda", "ceda", "ceda", transform=True)
    min = target_dataset.min(axis=0).to_dict()
    max = target_dataset.max(axis=0).to_dict()

    
    features = list(source_dataset.columns)
    features = ['temperature_1_5m', 'relative_humidity_1_5m', 'diffuse_surface_SW_flux', 'direct_surface_SW_flux', 'downward_surface_SW_flux', 'PoA', 'P_24h_shift', "is_day"]

    
    _,eval_obj = target(target_dataset, features, eval_dataset, trial, optimizer_name, lr_target,
                            n_layers_target, n_nodes_target, batch_size_target, [min,max])

    accuracy = eval_obj.metrics()['RMSE']

    return accuracy


if __name__ == "__main__":

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "No_feature_select"  # Unique identifier of the study.
    storage_name = "sqlite:///HP_target.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)
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

