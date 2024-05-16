import optuna
from optuna.trial import TrialState

df = pd.read_pickle("evaluation/source/metrics.pkl")
print(df)
