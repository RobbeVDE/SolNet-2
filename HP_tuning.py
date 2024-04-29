from optuna.trial import TrialState
from optuna import samplers 
from hyperparameters.hyperparameters import hyperparameters_source, hyperparameters_target
from Data.Featurisation import data_handeler
from scale import Scale
import torch
import logging
import sys
import pickle
import optuna
from functools import partial
installation_int = 0 # HP tuning is done with data for the first NL site
def HP_tuning(domain, model, step):
    source_state_dict = None
    TL = True
    match model:
            case 0:
                phys = False
                dataset_name = "nwp"           
            case 1:
                phys = True           
                dataset_name = "nwp"
            case 2:
                dataset_name = "no_weather"
                phys = False
            case 3:          
                phys = False
                dataset_name = "nwp"
                TL = False
            case 4:          
                phys = True
                dataset_name = "nwp"
                TL = False
            case 5:
                phys = False
                dataset_name = "era5"
            case 6:
                phys = True
                dataset_name ="era5"
    source_data, target_data, _ = data_handeler(installation_int, dataset_name, "nwp", "nwp", phys)
    if phys:
        phys_str = "phys.pkl"
    else:
        phys_str= "no_phys.pkl"    
    if domain == "source":
        from hyperparameters.OptunaSource import objective
        dataset = source_data
    elif domain == "target":
        from hyperparameters.OptunaTa import objective
        dataset = target_data
        if TL:  
            source_state_file = f"Models/source/{dataset_name}_{installation_int}_"
            source_state_file += phys_str          
            source_state_dict = torch.load(source_state_file)

    ftr_file = "hyperparameters/features_"
    if model == 2:#NO weather features
        ftr_file += "no_weather_"
    ftr_file += phys_str


    scale = Scale() #Load right scale
    scale.load(installation_int, dataset_name)

    objective = partial(objective,  dataset = dataset, source_state_dict = source_state_dict, scale=scale, step=step, case_n=model)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    storage_name = f"sqlite:///HP_{domain}.db"
    study_name = f"{dataset_name} | Physics: {phys} | TL: {TL} | Step: {step}"

    #Sampler initialisation
    try:
        sampler = pickle.load(open(f"hyperparameters/samplers/sampler_{domain}_{dataset_name}_{phys}_{TL}_{step}.pkl", "rb"))
        print("Using existing sampler.") 
    except: #If there is no sampler present, make a new one
        if step == 1: #Initial HP tuning is exploration based such that we know importance of HP better
            sampler = samplers.RandomSampler(seed=10)  # Make the sampler behave in a deterministic way. For reproducibility.
        else:        
            sampler = samplers.TPESampler(seed=10)
        print("Initialize a new sampler")

    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", 
                                load_if_exists=True, sampler=sampler)
    try:
        if step == 2:
            n_trials = 300
            if model == 2: # no weather cov so only 8 possib
                n_trials = 8
        elif step == 1:
            n_trials = 50
        else:
            n_trials = 100
        study.optimize(objective, n_trials=n_trials)
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        if step==2:
            final_features = []
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
                if value:
                    final_features.append(key)
                    if "_sin" in key:
                        key.replace("_sin","_cos") #Make sure both cyclical features are saved
                        final_features.append(key)
                    elif ("_cos" in key):
                        key.replace("_cos","_sin")
                        final_features.append(key)

            with open(ftr_file, 'wb') as f:
                pickle.dump(final_features, f)
            
        else:
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
                if "n_layer" in key:
                    n_layers = value
                elif "n_units" in key:
                    n_units = value
                elif "lr" in key:
                    lr = value
                elif "dropout" in key:
                    dropout = value
                elif "Batch_size" in key:
                    batch_size = value
                elif "optimizer" in key:
                    optimizer = value
                else:
                    print("This value not stored in hp object")
            if domain == "source":
                hp = hyperparameters_source(optimizer, lr, n_layers, n_units, dropout, batch_size)
            else:
                hp_source = hyperparameters_source()
                hp_source.load(model,3) #Load hyperparam source for n_layers and stuf
                hp = hyperparameters_target(hp_source.optimizer_name, lr, hp_source.n_layers, hp_source.n_nodes,
                                            dropout, batch_size) #Only parameters you optimized
            hp.save(model,step)
        with open(f"hyperparameters/samplers/sampler_{domain}_{dataset_name}_{phys}_{TL}_{step}.pkl", "wb") as fout: 
            pickle.dump(study.sampler, fout)
            print("Sampler saved succesfully.")
            
    except KeyboardInterrupt: #If optimization process gets interrupted the sampler is saved for next time 
        with open(f"hyperparameters/samplers/sampler_{domain}_{dataset_name}_{phys}_{TL}_{step}.pkl", "wb") as fout: 
            pickle.dump(study.sampler, fout)
        print("Sampler saved succesfully.")

if __name__ == "__main__":
    manual_enter = True
    if manual_enter:
        model = int(input("Specify model:\n 0. TL(no phys)               | 4. target(no S, phys)) |  \n 1. TL(phys)                  | 5. TL(era5, no phys)   |  \n 2. TL(no weather cov)        | 6. TL(era5, phys)      | 10. CNN-LSTM ?? \n 3. target(no S, no phys))    | 7. biLSTM              | \n"))
        domain = str(input("Domain: Enter source or target \n"))  # Unique identifier of the study.
        # dataset_name = str(input("Dataset: Enter nwp, era5 or no_weather \n"))
        # transfo = str(input("Use phys transfo: Enter True or False \n"))

        # if transfo in ["True", "true"]:
        #     transfo = True
        # elif transfo in ["False", "false"]:
        #     transfo = False
        # else:
        #     raise KeyError

        # TL = str(input("TL case: Enter True or False \n"))
        # if TL in ["True", "true"]:
        #     TL = True
        # elif TL in ["False", "false"]:
        #     TL = False
        # else:
        #     raise KeyError
        step = int(input("Select step in optimization: \n 1: Initial HP \n 2: Feature Selection \n 3: Complete HP \n"))
    else:
        domain = "source"
        dataset_name = "nwp"
        transfo = False
        TL = True
        step = 1
    HP_tuning(domain,model, step)
