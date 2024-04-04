from Data.Featurisation import Featurisation, data_handeler
from evaluation.evaluation import Evaluation
from main import TL_forecaster
source_dataset, target_dataset, eval_dataset = data_handeler("openmeteo", "ceda", "ceda", transform=True, month_source=True)

features = ['temperature_1_5m', 'relative_humidity_1_5m', 'diffuse_surface_SW_flux', 'direct_surface_SW_flux', 'downward_surface_SW_flux', 'PoA', 'P_24h_shift', "is_day"]
min = source_dataset.min(axis=0).to_dict()
max = source_dataset.max(axis=0).to_dict()

source_state_dict, target_state_dict, y_truth, y_forecast = TL_forecaster(source_dataset, target_dataset, features, eval_dataset, [min, max])

