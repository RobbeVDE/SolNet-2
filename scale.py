import pandas as pd
from Data.Featurisation import data_handeler
import pickle
class Scale:
    def __init__(self,
                 min = None,
                 max = None,
                 train_test_split=None) -> None:
        self.min = min
        self.max = max
        self.split = train_test_split
    def save(self,
             dataset_name,  
             path="hyperparameters/scale"):
        with open(f"{path}/{dataset_name}.pkl", 'wb') as f:
            pickle.dump(self, f)

    def load(self,
              dataset_name,
              path="hyperparameters/scale"):
        with open(f"{path}/{dataset_name}.pkl", 'rb') as f:
            old_scale = pickle.load(f)
            self.min = old_scale.min
            self.max = old_scale.max
            self.split = old_scale.split
    
    def calcul(self, dataset, train_split=0.8):
        train_len = int(train_split*len(dataset.index))
    
        self.min = dataset.iloc[:train_len,:].min(axis=0).to_dict()
        self.max = dataset.iloc[:train_len,:].max(axis=0).to_dict()
        self.split = train_split

        
        

if __name__ == "__main__":
    installation_id = "3437BD60"
    dataset_name = input("Dataset Name: Enter nwp, era5 or no_weather\n")
    source_dataset,_,_ = data_handeler(installation_id, dataset_name, dataset_name, dataset_name)
    scale = Scale()
    scale.calcul(source_dataset)
    scale.save(dataset_name)
    