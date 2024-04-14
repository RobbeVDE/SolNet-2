import pickle

class hyperparameters_source():
    def __init__(self,
                 optimizer_name=None,
                 lr=None,
                 n_layers=None,
                 n_nodes=None,
                 dropout=None,
                 batch_size=None,
                 trial = None):
        self.trial = trial
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.batch_size = batch_size
    
    def save(self, step):
        with open(f"hyperparameters/HP_source{step}.pkl", 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, step):
        with open(f"hyperparameters/HP_source{step}.pkl", 'rb') as f:
            hp = pickle.load(f)
            self.optimizer_name = hp.optimizer_name
            self.lr = hp.lr
            self.n_layers = hp.n_layers
            self.n_nodes = hp.n_nodes
            self.dropout = hp.dropout
            self.batch_size = hp.batch_size

class hyperparameters_target():
    def __init__(self,
                 optimizer_name=None,
                 lr= None,
                 n_layers= None,
                 n_nodes= None,
                 dropout=None,
                 batch_size=None,
                 trial = None,
                 source_state_dict = None):
        self.trial = trial
        self.optimizer_name = optimizer_name
        self.source_state_dict = source_state_dict 
        self.lr = lr
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.batch_size = batch_size

    def save(self, step):
        with open(f"hyperparameters/HP_target{step}.pkl", 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, step=1):
        with open(f"hyperparameters/HP_target{step}.pkl", 'rb') as f:
            hp = pickle.load(f)
            self.optimizer_name = hp.optimizer_name
            self.lr = hp.lr
            self.n_layers = hp.n_layers
            self.n_nodes = hp.n_nodes
            self.dropout = hp.dropout
            self.batch_size = hp.batch_size
