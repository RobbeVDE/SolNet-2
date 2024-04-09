import pandas as pd

dataset = pd.read_pickle("Data/is_day.pickle")
dataset.to_csv("Data/is_day.csv", index_label=False)


