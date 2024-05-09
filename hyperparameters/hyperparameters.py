import pickle

class hyperparameters_source():
    def __init__(self,
                 optimizer_name=None,
                 lr=None,
                 n_layers=None,
                 n_nodes=None,
                 dropout=None,
                 batch_size=None,
                 wd=None,
                 trial = None,
                 bidirectional=False,
                 gif_plotter=False):
        self.trial = trial
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.wd = wd
        self.bd = bidirectional
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.batch_size = batch_size
        self.gif_plotter = gif_plotter
    
    def save(self, case_n):
        with open(f"hyperparameters/HP_source_{case_n}.pkl", 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, case_n):
        with open(f"hyperparameters/HP_source_{case_n}.pkl", 'rb') as f:
            hp = pickle.load(f)
            self.optimizer_name = hp.optimizer_name
            self.wd = hp.wd
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
                 wd=None,
                 trial = None,
                 bidirectional=False,
                 source_state_dict = None):
        self.trial = trial
        self.optimizer_name = optimizer_name
        self.source_state_dict = source_state_dict 
        self.lr = lr
        self.wd = wd
        self.n_layers = n_layers
        self.bd = bidirectional
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.batch_size = batch_size

    def save(self, case_n):
        with open(f"hyperparameters/HP_target_{case_n}.pkl", 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, case_n):
        with open(f"hyperparameters/HP_target_{case_n}.pkl", 'rb') as f:
            hp = pickle.load(f)
            self.optimizer_name = hp.optimizer_name
            self.lr = hp.lr
            self.wd = hp.wd
            self.n_layers = hp.n_layers
            self.n_nodes = hp.n_nodes
            self.dropout = hp.dropout
            self.batch_size = hp.batch_size


if __name__ == "__main__":
    model=0
    custom = False
    


    if custom:
        optimizer = "Adam"   
        dropout = 0.3393
        n_layers=3
        n_units = 387 
        
        wd = 4.75742e-7
        lr=0.000154969
        batch_size = 36
        hp = hyperparameters_source(optimizer, lr, n_layers, n_units, dropout, batch_size, wd)
        hp.save(model)

        wd = 1.24225e-8
        lr= 1.473e66-6
        batch_size = 54
        hp = hyperparameters_target(optimizer, lr, n_layers, n_units, dropout, batch_size, wd)
        hp.save(model)
    else:
        import optuna
        domain = "target"
        storage_name = f"sqlite:///HP_{domain}.db"
        study_name = f"nwp | Physics: False | TL: True"

        study = optuna.load_study(study_name=study_name, storage=storage_name)
        print(study)
        
        params = study.best_params

        for key, value in params.items():
            print("    {}: {}".format(key, value))
            if "n_layer" in key:
                n_layers = value
            elif "n_units" in key:
                n_units = value
            elif "lr" in key:
                lr = value
            elif "Weight_decay" in key:
                wd = value
            elif "dropout" in key:
                dropout = value
            elif "Batch_size" in key:
                batch_size = value
            elif "optimizer" in key:
                optimizer = value
            else:
                print("This value not stored in hp object")
        if domain == "source":
            hp = hyperparameters_source(optimizer, lr, n_layers, n_units, dropout, batch_size, wd)
        else:
            hp_source = hyperparameters_source()
            hp_source.load(model) #Load hyperparam source for n_layers and stuf
            hp = hyperparameters_target(hp_source.optimizer_name, lr, hp_source.n_layers, hp_source.n_nodes,
                                        hp_source.dropout, batch_size, wd) #Only parameters you optimized
        hp.save(model)
        
    