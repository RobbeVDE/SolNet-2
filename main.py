import pickle
import pandas as pd
from tensors.Tensorisation import Tensorisation
import torch
from Models.training import Training
from Models.lstm import LSTM

def source_trainer(dataset, lags, forecast_period, input_size, hidden_size, num_layers, dropout, device):

    ## Make sure that dataset is just a dataframe and not a list of dataframes
    # Get the list of features

    features = list(dataset.columns)
    tensors = Tensorisation(dataset, 'P', features, lags, forecast_period)
    X_train, X_test, y_train, y_test = tensors.tensor_creation()
    
    print("Shape of data: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    

    my_lstm = LSTM(input_size,hidden_size,num_layers, forecast_period, dropout).to(device)
    print(my_lstm)

    # Set the training parameters
    epochs = 80

    # Initialize the trainer
    training = Training(my_lstm, X_train, y_train, X_test, y_test, epochs)

    # Train the model and return the trained parameters and the best iteration
    state_dict_list, best_epoch = training.fit()

    return my_lstm, state_dict_list, best_epoch

def target_trainer(model, dataset, lags, forecast_period, input_size, hidden_size, num_layers, dropout, device):
     ## Make sure that dataset is just a dataframe and not a list of dataframes
    # Get the list of features

    features = list(dataset.columns)
    tensors = Tensorisation(dataset, 'P', features, lags, forecast_period)
    X_train, X_test, y_train, y_test = tensors.tensor_creation()
    
    print("Shape of data: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
   

    my_lstm = LSTM(input_size,hidden_size,num_layers, forecast_period, dropout).to(device)
    print(my_lstm)
    
    # Set the training parameters
    epochs = 80

    # Initialize the trainer
    training = Training(my_lstm, X_train, y_train, X_test, y_test, epochs)

    # Train the model and return the trained parameters and the best iteration
    state_dict_list, best_epoch = training.fit()

    return my_lstm, state_dict_list, best_epoch

def forecast_maker(source_data, target_data, test_data, y_name,hyper_tuning, transposition, tuning_method):

    #### SOURCE MODEL ########
    lags = 24
    forecast_period = 24
    # Set the parameters for the lstm
    input_size = len(features)
    hidden_size = 100
    num_layers = 4
    dropout = 0.3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_lstm, source_state_dict, source_epoch = source_trainer(source_data, lags, forecast_period, input_size, hidden_size, num_layers, dropout, device)
    

    #### TRANSFER MODEL #####
    transfer_model = LSTM(input_size,hidden_size,num_layers, forecast_period, dropout).to(device)
    transfer_model = transfer_model.load_state_dict(source_state_dict)
    if tuning_method == 'freeze':
        #Freeze all layers except output layer
        
    elif tuning_method == "whole":
        #Change all parameters but with really low learning rated
        pass
    else:
        raise KeyError
    target_lstm, target_state_dict, target_epoch = target_trainer(source_lstm, target_data, lags, forecast_period)

    
    return state_dict_list, best_epoch,

def target_renamer(dataset, original_name):
    #Rename column with target to 'P' to simplify rest of code
    dataset = dataset.rename(columns ={original_name:'P'})
    return dataset
