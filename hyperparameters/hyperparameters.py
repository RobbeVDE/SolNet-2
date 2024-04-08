class hyperparameters_source():
    def __init__(self,
                 trial,
                 optimizer_name,
                 lr,
                 n_layers,
                 n_nodes,
                 dropout,
                 batch_size):
        self.trial
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.batch_size = batch_size

class hyperparameters_target():
    def __init__(self,
                 trial,
                 lr,
                 n_layers,
                 n_nodes,
                 dropout,
                 batch_size):
        self.trial = trial
        self.lr = lr
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.batch_size = batch_size