import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example data
data = {
    ('Location1', '2023-01'): [0.8, 0.75, 0.77, 0.79, 0.76, 0.78],
    ('Location1', '2023-02'): [0.82, 0.80, 0.81, 0.79, 0.85, 0.83],
    ('Location1', '2023-03'): [0.84, 0.82, 0.85, 0.83, 0.87, 0.86],
    ('Location2', '2023-01'): [0.65, 0.67, 0.66, 0.68, 0.70, 0.69],
    ('Location2', '2023-02'): [0.72, 0.70, 0.73, 0.71, 0.75, 0.74],
    ('Location2', '2023-03'): [0.73, 0.72, 0.74, 0.75, 0.77, 0.76]
}
index = pd.MultiIndex.from_tuples(data.keys(), names=['Location', 'Month'])
df = pd.DataFrame(data.values(), index=index, columns=['Model1', 'Model2', 'Model3', 'Model4', 'Model5', 'Model6'])


