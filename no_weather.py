from Data.Featurisation import Featurisation, data_handeler
from evaluation.evaluation import Evaluation
from main import TL_forecaster
source_dataset, target_dataset, eval_dataset = data_handeler("ceda", "ceda", "ceda", transform=True, month_source=False)

features = ['P_24h_shift', "month_sin", "month_cos", "hour_sin", "hour_cos"]
min = source_dataset.min(axis=0).to_dict()
max = source_dataset.max(axis=0).to_dict()

source_state_dict, target_state_dict, y_truth, y_forecast = TL_forecaster(source_dataset, target_dataset, features, eval_dataset, [min, max])


