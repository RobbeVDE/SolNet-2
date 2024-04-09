from optuna.trial import TrialState
from optuna.samplers import TPESampler
from hyperparameters.hyperparameters import hyperparameters_source, hyperparameters_target
from Data.Featurisation import data_handeler
from scale import Scale
import torch
import logging
import sys
import pickle
import optuna
from functools import partial


def HP_tuning(tuning_model, dataset_name, transfo, TL, step):
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

    scale = Scale() #Load right scale
    scale.load(dataset_name)

    objective = partial(objective,  dataset = dataset, source_state_dict = source_state_dict, scale=scale, step=step)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    storage_name = f"sqlite:///HP_{tuning_model}.db"
    study_name = f"{dataset_name} | transfo: {transfo} | TL: {TL} | Step: {step}"
    try:
        sampler = pickle.load(open(f"hyperparameters/samplers/sampler_{tuning_model}_{dataset_name}_{transfo}_{TL}_{step}.pkl", "rb")) 
    except: #If there is no sampler present, make a new one
        sampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.

    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", 
                                load_if_exists=True, sampler=sampler)
    try:
        if step == 1: #Less trials for fist estimate at HP bcs not that important yet
            n_trials = 30
        else:
            n_trials = 100
        study.optimize(objective, n_trials=n_trials)
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
        if step==2:
            final_features = []
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
                if value:
                    final_features.append(key)

            with open(f"hyperparameters/HP_{tuning_model}.pkl", 'wb') as f:
                pickle.dump(final_features, f)
            
        else:
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
                if "n_layer" in key:
                    n_layers = value
                elif "n_units" in key:
                    n_units = value
                elif "lr" in key:
                    lr = value
                elif "dropout" in key:
                    dropout = value
                elif "Batch_size" in key:
                    batch_size = value
                elif "optimizer" in key:
                    optimizer = value
                else:
                    print("This value not stored in hp object")
                if tuning_model == "source":
                    hp = hyperparameters_source(optimizer, lr, n_layers, n_units, dropout, batch_size)
                else:
                    hp_source = hyperparameters_source()
                    hp_source.load(3) #Load hyperparam source for n_layers and stuf
                    hp = hyperparameters_target(hp_source.optimizer_name, lr, hp_source.n_layers, hp_source.n_nodes,
                                                dropout, batch_size) #Only parameters you optimized
                hp.save(step)

            
    except KeyboardInterrupt: #If optimization process gets interrupted the sampler is saved for next time 
        with open(f"hyperparameters/samplers/sampler_{tuning_model}_{dataset_name}_{transfo}_{TL}_{step}.pkl", "wb") as fout: 
            pickle.dump(study.sampler, fout)

if __name__ == "__main__":
    manual_enter = True
    if manual_enter:
        tuning_model = str(input("Tuning Model: Enter source or target \n"))  # Unique identifier of the study.
        dataset_name = str(input("Dataset: Enter nwp or era5 \n"))
        transfo = input("Use phys transfo: Enter True or False \n")

        if transfo in ["True", "true"]:
            transfo = True
        elif transfo in ["False", "false"]:
            transfo = False
        else:
            raise KeyError

        TL = bool(input("TL case: Enter True or False \n"))
        if TL in ["True", "true"]:
            TL = True
        elif TL in ["False", "false"]:
            TL = False
        else:
            raise KeyError
        step = int(input("Select step in optimization: \n 1: Initial HP \n 2: Feature Selection \n 3: Complete HP \n"))
    else:
        tuning_model = "source"
        dataset_name = "nwp"
        transfo = False
        TL = True
        step = 1
    HP_tuning(tuning_model, dataset_name, transfo, TL, step)