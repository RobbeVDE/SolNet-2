from Models.models import source
from Data.Featurisation import data_handeler
from hyperparameters.hyperparameters import hyperparameters_source
from evaluation.metric_processor import metric_processor_source
from scale import Scale
import torch
import pickle
import os.path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ctn_eval = True #Loop trough all models and sites
if ctn_eval:
    models = range(9)
    sites = range(9)
    HP_tuning = False
    if HP_tuning:
        sites = [0]
else:
    models = [int(input("Specify model:\n 0. TL(no phys)               | 4. target(no S, phys)) | 8. physical \n 1. TL(phys)                  | 5. TL(era5, no phys)   | 9. persistence \n 2. TL(no weather cov)        | 6. TL(era5, phys)      | 10. CNN-LSTM ?? \n 3. target(no S, no phys))    | 7. biLSTM              | 11. ARIMA ?? \n"))]
    sites = [int(input("Specify site: \n 0. NL 1       | 3. UK \n 1. NL 2        |   \n 2. Costa Rica  | \n"))]
    HP_tuning = False 
TL = True
for model in models:
    for installation_int in sites:
        print(f"Currently training model: {model}, for site {installation_int}")
        match model:
                case 0:
                    phys = False
                    dataset_name = "nwp" 
                    TL = True          
                case 1:
                    phys = True           
                    dataset_name = "nwp"
                    TL = True
                case 2:
                    dataset_name = "no_weather"
                    phys = False
                    TL = True
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
                    TL = True
                case 6:
                    phys = True
                    dataset_name ="era5"
                    TL = True
                case 7:
                    TL = False # Will stop the iteration
        if TL:
            source_data, target_data, _ = data_handeler(installation_int, dataset_name, "nwp", "nwp", phys, HP_tuning=HP_tuning, decomp=True)
            if phys:
                phys_str = "phys.pkl"
            else:
                phys_str= "no_phys.pkl"    


            ftr_file = "features/ft_"
            if model == 2:#NO weather features
                ftr_file += "no_weather_"
            ftr_file += phys_str


            if os.path.isfile(ftr_file):
                with open(ftr_file, 'rb') as f:
                    features = pickle.load(f)
            else:
                features = ['temperature_1_5m', 'relative_humidity_1_5m', 'diffuse_surface_SW_flux', 'direct_surface_SW_flux', 'downward_surface_SW_flux', 'P_24h_shift', 'total_cloud_amount']

            scale = Scale()
            scale.load(installation_int, dataset_name, phys)

            hp = hyperparameters_source()
            try:
                hp.load(model)
            except:
                hp.load(0)
            hp.gif_plotter = False
            hp.bd =False
            accuracy, state_dict, timer = source(source_data, features, hp, scale)
            state_dict = state_dict
            metric_processor_source(accuracy, timer, scale, model, installation_int)

            torch.save(state_dict, f"Models/source/{dataset_name}_{installation_int}_{phys_str}")
        else:
            print("No source model for this case")
