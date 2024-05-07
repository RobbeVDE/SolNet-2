import optuna
import os

model = int(input("Specify model:\n 0. TL(no phys)               | 4. target(no S, phys)) |  \n 1. TL(phys)                  | 5. TL(era5, no phys)   |  \n 2. TL(no weather cov)        | 6. TL(era5, phys)      | 10. CNN-LSTM ?? \n 3. target(no S, no phys))    | 7. biLSTM              | \n"))
domain = str(input("Domain: Enter source or target \n"))  # Unique identifier of the study.
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
if phys:
    phys_str = "phys.pkl"
else:
    phys_str= "no_phys.pkl"    
#Remove study object
storage_name = f"sqlite:///HP_{domain}.db"
study_name = f"{dataset_name} | Physics: {phys} | TL: {TL}"
    
optuna.delete_study(study_name=study_name, storage=storage_name)



#Remove sampler
filepath = f"hyperparameters/samplers/sampler_{domain}_{dataset_name}_{phys}_{TL}.pkl"
if os.path.exists(filepath):
  os.remove(filepath)
