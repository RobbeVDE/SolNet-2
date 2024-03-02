import pickle
import pandas as pd

#1 Get all data in
#with open('Data/Rebase.pickle', 'rb') as f:
#    rebase = pickle.load(f)

openmeteo = pd.read_pickle("Data/openmeteo.pickle")

pvgis = pd.read_csv("Data/PVGIS demo data.csv", sep=',')
pvgis.index = pd.to_datetime(pvgis['time'], utc=True,  format='%Y%m%d:%H%M',yearfirst=True) - pd.Timedelta(minutes=10)
pvgis.drop('time', axis=1, inplace=True)

pv_power = pvgis.xs('P', axis=1)
pv_power = pv_power['2020-08-01': '2020-08-31']

data = pd.merge(pv_power, openmeteo, left_index=True, right_index=True)


#2. Featurise data & ready for training & testing 7
from Data.Featurisation import Featurisation
print(Featurisation(data))
 