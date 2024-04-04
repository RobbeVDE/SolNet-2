from Data.Featurisation import Featurisation, data_handeler
from evaluation.evaluation import Evaluation

source_dataset, target_dataset, eval_dataset = data_handeler("ceda", "ceda", "ceda", transform=True, month_source=False)

#Persistence is just using the P_24h shift

y_forecast = eval_dataset['P_24h_shift']
y_truth = eval_dataset['P']
eval_obj = Evaluation(y_truth, y_forecast)
print(eval_obj.metrics())

