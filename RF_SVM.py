import pandas as pd
from Data.Featurisation import data_handeler
from Models.models import source, target
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
from scale import Scale
from hyperparameters.hyperparameters import hyperparameters_source, hyperparameters_target
import torch
import scienceplots
plt.style.use(['science', 'notebook'])
import warnings
import numpy as np
from sklearn.svm import SVR
from Models.lstm import LSTM
from tensors.Tensorisation import Tensorisation
from scale import Scale
from pvlib import location

rmse_rf = pd.DataFrame(index=range(4), columns=(range(9)))
metadata = pd.read_pickle("Data/Sites/metadata.pkl")
for site in range(9):
    for model in range(4): #Only treat nwp vs reanalysis and physics vs non-physics
        if model in [0,2]:
            phys=False
            phys_str = "no_phys.pkl"
        else:
            phys=True
            phys_str = "phys.pkl"
        if model in [0,1]:
            dataset_name = "nwp"
        else:
            dataset_name="era5"
        source_data,_,eval_data = data_handeler(site, dataset_name, "nwp", "nwp", phys, HP_tuning=False)
        eval_data = eval_data[24:31*24] #Only assess the first month
        lat = metadata.loc[site, "Latitude"]
        lon= metadata.loc[site, "Longitude"]
        loc = location.Location(lat, lon,altitude=location.lookup_altitude(lat,lon))
        source_times = source_data.index
        eval_times = eval_data.index

        sol_pos_source = loc.get_solarposition(source_times)
        sol_pos_eval = loc.get_solarposition(eval_times)

        zenith_filter_source = sol_pos_source["zenith"] <= 85
        zenith_filter_eval = sol_pos_eval["zenith"] <= 85

        source_data = source_data[zenith_filter_source]
        eval_data = eval_data[zenith_filter_eval]
        # Labels are the values we want to predict
        labels = np.array(source_data['P'])
        # Remove the labels from the source_data
        # axis 1 refers to the columns
        source_data= source_data.drop('P', axis = 1)
        # Saving feature names for later use
        ftr_file = "features/ft_" + phys_str
        if os.path.isfile(ftr_file):
            with open(ftr_file, 'rb') as f:
                feature_list = pickle.load(f)
        # Convert to numpy array
        source_data = source_data[feature_list]
        source_data = np.array(source_data)

        # Split the data into training and testing sets
        train_features, test_features, train_labels, test_labels = train_test_split(source_data, labels, test_size = 0.25, random_state = 42)

        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', test_labels.shape)
        #Instantiate model with 1000 decision trees
        rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        # Train the model on training data
        rf.fit(train_features, train_labels)

        # Use the forest's predict method on the test data
        predictions = rf.predict(test_features)
        # Calculate the absolute errors
        errors = np.sqrt(np.mean(np.square(predictions - test_labels)))
        # Print out rmse
        print("Test Error: ", errors, 'W')


        # Use the forest's predict method on the test data
        val_labels = eval_data['P']
        val_features = eval_data[feature_list]
        forecasts = rf.predict(val_features)
        # Calculate the absolute errors
        errors = np.sqrt(np.mean(np.square(forecasts - val_labels)))
        # Print out rmse
        print("Zero-shot RMSE: ", errors, 'W')
        rmse_rf.loc[model, site] = errors
rmse_rf.to_pickle("sensitivity_analysis/rmse_rf.pkl")

rmse_svm = pd.DataFrame(index=range(4), columns=(range(9)))

for site in range(9):
    for model in range(4): #Only treat nwp vs reanalysis and physics vs non-physics
        if model in [0,2]:
            phys=False
            phys_str = "no_phys.pkl"
        else:
            phys=True
            phys_str = "phys.pkl"
        if model in [0,1]:
            dataset_name = "nwp"
        else:
            dataset_name="era5"
        source_data,_,eval_data = data_handeler(site, dataset_name, "nwp", "nwp", phys, HP_tuning=False)
        source_data,_,eval_data = data_handeler(site, dataset_name, "nwp", "nwp", phys, HP_tuning=False)
        eval_data = eval_data[24:31*24] #Only assess the first month
        lat = metadata.loc[site, "Latitude"]
        lon= metadata.loc[site, "Longitude"]
        loc = location.Location(lat, lon,altitude=location.lookup_altitude(lat,lon))
        source_times = source_data.index
        eval_times = eval_data.index

        sol_pos_source = loc.get_solarposition(source_times)
        sol_pos_eval = loc.get_solarposition(eval_times)

        zenith_filter_source = sol_pos_source["zenith"] <= 85
        zenith_filter_eval = sol_pos_eval["zenith"] <= 85

        source_data = source_data[zenith_filter_source]
        eval_data = eval_data[zenith_filter_eval]
        #Scale data (necessary for SVR)
        scale = Scale()
        scale.load(site, dataset_name, phys)
        max = scale.max
        min = scale.min
        for covar in source_data.columns:
            source_data[covar] = (source_data[covar]-min[covar])/(max[covar]-min[covar])
            eval_data[covar] = (eval_data[covar]-min[covar])/(max[covar]-min[covar])

        # Labels are the values we want to predict
        labels = np.array(source_data['P'])
        # Remove the labels from the source_data
        # axis 1 refers to the columns
        source_data= source_data.drop('P', axis = 1)
        # Saving feature names for later use
        ftr_file = "features/ft_" + phys_str
        if os.path.isfile(ftr_file):
            with open(ftr_file, 'rb') as f:
                feature_list = pickle.load(f)
        # Convert to numpy array
        source_data = source_data[feature_list]
        source_data = np.array(source_data)


        # Split the data into training and testing sets
        train_features, test_features, train_labels, test_labels = train_test_split(source_data, labels, test_size = 0.25, random_state = 42)

        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', test_labels.shape)
        



        #Instantiate model SVR
        rf = SVR(kernel='linear')
        # Train the model on training data
        rf.fit(train_features, train_labels)

        # Use the forest's predict method on the test data
        predictions = rf.predict(test_features)
        # Calculate the absolute errors
        errors = np.sqrt(np.mean(np.square(predictions - test_labels)))
        # Print out rmse
        print("Test Error: ", errors, 'W')


        # Use the forest's predict method on the test data
        val_labels = eval_data['P']
        val_features = eval_data[feature_list]
        forecasts = rf.predict(val_features)
        # Calculate the absolute errors
        errors = np.sqrt(np.mean(np.square(forecasts - val_labels)))
        # Print out rmse
        print("Zero-shot RMSE: ", errors*(max['P']-min['P']), 'W')
        rmse_svm.loc[model, site] = errors*(max['P']-min['P'])

rmse_svm.to_pickle("sensitivity_analysis/rmse_svm.pkl")
