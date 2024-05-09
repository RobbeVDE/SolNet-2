from hyperparameters.hyperparameters import hyperparameters_target
import pickle
with open(f"hyperparameters/HP_target_{0}.pkl", 'rb') as f:
    hp = pickle.load(f)
hp.wd =5e-8
hp.save(0)