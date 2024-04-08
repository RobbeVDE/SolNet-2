from optuna.trial import TrialState
import logging
import sys
import optuna
study_name = "NWP"  # Unique identifier of the study.
if study_name == "Source NWP":
    from hyperparameters.OptunaSource import objective
elif study_name == "Source hist weather":
    from hyperparameters.OptunaSourceHW import objective
elif study_name == "Target":
    from hyperparameters.OptunaTa import objective
elif study_name == "Target (no Source)":
    from hyperparameters.OptunaTa import objective_no_source


if __name__ == "__main__":

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    storage_name = "sqlite:///HP_TL.db"
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