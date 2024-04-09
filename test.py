from hyperparameters.hyperparameters import hyperparameters_source
batch_size = 4
lr = 1.75e-3
dropout= 0.11
n_layers = 2
n_nodes = 190
optimizer_name = "RMSprop"
hp = hyperparameters_source(optimizer_name, lr, n_layers, n_nodes, dropout, batch_size)
hp.save(1)


