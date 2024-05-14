import pandas as pd
from Data.Featurisation import data_handeler
import pickle
class Scale:
    def __init__(self,
                 min = None,
                 max = None) -> None:
        self.min = min
        self.max = max
    def save(self,
             site,
             dataset_name,
             phys,  
             path="hyperparameters/scale"):
        with open(f"{path}/{dataset_name}_{site}_{phys}.pkl", 'wb') as f:
            pickle.dump(self, f)

    def load(self,
             site,
             dataset_name,
             phys,
              path="hyperparameters/scale"):
        with open(f"{path}/{dataset_name}_{site}_{phys}.pkl", 'rb') as f:
            old_scale = pickle.load(f)
            self.min = old_scale.min
            self.max = old_scale.max
    
    def calcul(self, dataset):
    
        self.min = dataset.min(axis=0).to_dict()
        self.max = dataset.max(axis=0).to_dict()

        
        

if __name__ == "__main__":
    ctn_eval = True #Loop trough all models and sites
    if ctn_eval:
        models = range(9)
        sites = list(range(4))
        sites.remove(1)
    else:
        models = [int(input("Specify model:\n 0. TL(no phys)               | 4. target(no S, phys)) | 8. physical \n 1. TL(phys)                  | 5. TL(era5, no phys)   | 9. persistence \n 2. TL(no weather cov)        | 6. TL(era5, phys)      | 10. CNN-LSTM ?? \n 3. target(no S, no phys))    | 7. biLSTM              | 11. ARIMA ?? \n"))]
        sites = [int(input("Specify site: \n 0. NL 1       | 3. UK \n 1. NL 2        |   \n 2. Costa Rica  | \n"))]
        
    TL = True
    for model in models:
        for installation_int in sites:
    
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
            source_dataset, _, _ = data_handeler(installation_int, dataset_name, "nwp", "nwp", phys)
            scale = Scale()
            scale.calcul(source_dataset)
            print(scale.max)
            scale.save(installation_int, dataset_name, phys)
    