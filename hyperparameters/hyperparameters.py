class hyperparameters_source():
    def __init__(self,
                 optimizer_name,
                 lr,
                 n_layers,
                 n_nodes,
                 dropout,
                 batch_size,
                 trial):
        self.trial = trial
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.batch_size = batch_size

class hyperparameters_target():
    def __init__(self,
                 lr,
                 n_layers,
                 n_nodes,
                 dropout,
                 batch_size,
                 trial = None,
                 source_state_dict = None):
        self.trial = trial
        self.source_state_dict = source_state_dict 
        self.lr = lr
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.batch_size = batch_size