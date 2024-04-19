from hyperparameters.hyperparameters import hyperparameters_source
batch_size = 40
lr = 1e-4
dropout= 0.31
n_layers = 3
n_nodes = 454
optimizer_name = "Adam"
hp = hyperparameters_source(optimizer_name, lr, n_layers, n_nodes, dropout, batch_size)
hp.save(2,1)


