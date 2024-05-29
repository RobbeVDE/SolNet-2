from pvlib import irradiance, location
import pandas as pd
from Data.Featurisation import data_handeler

sourcce,_,_ = data_handeler(0, "nwp", "nwp", "nwp")
print(sourcce["P"].corr(sourcce["P_24h_shift"]))