import pandas as pd

dataset = pd.read_pickle("Data/PVGIS.pickle")
dataset.to_csv("Data/PVGIS.csv", index_label=False)


