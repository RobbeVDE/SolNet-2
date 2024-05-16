import optuna
from optuna.trial import TrialState

study = optuna.load_study(study_name="era5 | Physics: True | TL: True", storage="sqlite:///HP_target.db")

print(len(study.get_trials(deepcopy=False, states=[TrialState.PRUNED])))
