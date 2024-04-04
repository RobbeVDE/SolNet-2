from Data.Featurisation import Featurisation, data_handeler

source_dataset, target_dataset, eval_dataset = data_handeler("ceda", "ceda", "ceda", transform=True, month_source=False)

print(source_dataset)
print(target_dataset)
print(eval_dataset)

