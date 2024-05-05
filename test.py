from hyperparameters import hyperparameters

hp = hyperparameters.hyperparameters_target()
hp.load(0,3)
print(hp.dropout)

hp_s = hyperparameters.hyperparameters_source()
hp_s.load(0,3)
print(hp_s.dropout)

hp.dropout = hp_s.dropout
hp.save(0,3)
print(hp.dropout)