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
def HP_tuning(domain, model):
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
        patience = 5
    elif domain == "target":
        from hyperparameters.OptunaTa import objective
        dataset = target_data
        patience = 2
        if TL:  
            source_state_file = f"Models/source/{dataset_name}_{installation_int}_"
            source_state_file += phys_str          
            source_state_dict = torch.load(source_state_file)

    ftr_file = "hyperparameters/features_"
    if model == 2:#NO weather features
        ftr_file += "no_weather_"
    ftr_file += phys_str


    scale = Scale() #Load right scale
    scale.load(installation_int, dataset_name, phys)

    objective = partial(objective,  dataset = dataset, source_state_dict = source_state_dict, scale=scale, case_n=model)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    storage_name = f"sqlite:///HP_{domain}.db"
    study_name = f"{dataset_name} | Physics: {phys} | TL: {TL}"

    #Sampler initialisation
    try:
        sampler = pickle.load(open(f"hyperparameters/samplers/sampler_{domain}_{dataset_name}_{phys}_{TL}.pkl", "rb"))
        print("Using existing sampler.") 
    except: #If there is no sampler present, make a new one
        sampler = samplers.RandomSampler(seed=39)
        print("Initialize a new sampler")

    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", 
                                load_if_exists=True, sampler=sampler, pruner=optuna.pruners.MedianPruner(n_warmup_steps=patience))
    try:        
        study.optimize(objective, n_trials=500)
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
        
        for key, value in trial.params.items():
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
        n_layers = 1
        optimizer = "Adam"
        dropout = 0
        if domain == "source":
            hp = hyperparameters_source(optimizer, lr, n_layers, n_units, dropout, batch_size, wd)
        else:
            if model in [3,4]:
                hp = hyperparameters_target(optimizer, lr, n_layers, n_units, dropout, batch_size, wd)
            else:
                hp_source = hyperparameters_source()
                hp_source.load(model) #Load hyperparam source for n_layers and stuf
                hp = hyperparameters_target(hp_source.optimizer_name, lr, hp_source.n_layers, hp_source.n_nodes,
                                        hp_source.dropout, batch_size, wd) #Only parameters you optimized
        hp.save(model)
        with open(f"hyperparameters/samplers/sampler_{domain}_{dataset_name}_{phys}_{TL}.pkl", "wb") as fout: 
            pickle.dump(study.sampler, fout)
            print("Sampler saved succesfully.")
            
    except KeyboardInterrupt: #If optimization process gets interrupted the sampler is saved for next time 
        with open(f"hyperparameters/samplers/sampler_{domain}_{dataset_name}_{phys}_{TL}.pkl", "wb") as fout: 
            pickle.dump(study.sampler, fout)
        print("Sampler saved succesfully.")

if __name__ == "__main__":
    manual_enter = True

    if manual_enter:
        model = int(input("Specify model:\n 0. TL(no phys)               | 4. target(no S, phys)) |  \n 1. TL(phys)                  | 5. TL(era5, no phys)   |  \n 2. TL(no weather cov)        | 6. TL(era5, phys)      | 10. CNN-LSTM ?? \n 3. target(no S, no phys))    | 7. biLSTM              | \n"))
        domain = str(input("Domain: Enter source or target \n"))  # Unique identifier of the study.
        HP_tuning(domain, model)
    else:
        model_list = [2,5,6]
        domain = "source"
        for model in model_list:
            HP_tuning(domain,model)
