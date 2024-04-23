from Models.models import source
from Data.Featurisation import data_handeler
from hyperparameters.hyperparameters import hyperparameters_source
from scale import Scale
import torch
import pickle
#### Model parameters
installation_int = int(input("Specify site: \n 0. NL 1       | 3. UK \n 1. NL 2       \n 2. Costa Rica   \n"))
model = int(input("Specify model:\n 0. TL(no phys)               | 4. target(no S, phys)) |  \n 1. TL(phys)                  | 5. TL(era5, no phys)   |  \n 2. TL(no weather cov)        | 6. TL(era5, phys)      | 10. CNN-LSTM ?? \n 3. target(no S, no phys))    | 7. biLSTM              | \n"))
        
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


ftr_file = "hyperparameters/features_"
if model == 2:#NO weather features
    ftr_file += "no_weather_"
ftr_file += phys_str



with open(ftr_file, 'rb') as f:
            features = pickle.load(f)


scale = Scale()
scale.load(dataset_name)

hp = hyperparameters_source()
hp.load(model,3)
hp.gif_plotter = False
hp.bd =False
accuracy, state_dict = source(source_data, features, hp, scale)

torch.save(state_dict, f"Models/source_{dataset_name}_{phys_str}")
