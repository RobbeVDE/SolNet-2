import torch
import optuna
from main import forecast_maker, data_slicer
import pandas as pd
from Data.Featurisation import Featurisation
from optuna.trial import TrialState
import logging
import sys
installation_id = "3437BD60"
def objective(trial):

    # Generate the optimizersa and hyperparameters
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr_source = trial.suggest_loguniform("lr_source", 1e-5, 1e-1)
    lr_target = trial.suggest_loguniform("lr_target", 1e-5, 1e-1)

    n_layers_source = trial.suggest_int("n_layers_source", 1, 7)
    n_layers_target = trial.suggest_int("n_layers_target", 1, 7)

    n_nodes_source = trial.suggest_int("n_units_source",4,1024)
    n_nodes_target = trial.suggest_int("n_units_target",4,1024)

    dropout = trial.suggest_uniform("dropout_l", 0.1, 0.5)

    batch_size_source = trial.suggest_int("Batch_size_source", 4,64)
    batch_size_target = trial.suggest_int("Batch_size_target", 1,64)
    # Get data

    source_range = pd.date_range("2016-05-01","2020-07-31 23:00", freq='h', tz="UTC")
    target_range = pd.date_range("2020-08-01", "2020-08-31 23:00", freq='h', tz="UTC")
    eval_range = pd.date_range("2020-09-01", "2021-07-31 23:00", tz="UTC", freq='h')

    pvgis = pd.read_pickle('Data/PVGIS.pickle')

    CEDA = pd.read_pickle("CEDA_dataNL.pickle")
    is_day = pd.read_pickle("Data/is_day.pickle")

    power = pd.read_pickle("Data/NL_power.pickle")
    metadata = pd.read_csv("Data/installations Netherlands.csv", sep=';')
    metadata = metadata.set_index('id')
    metadata_id = metadata.loc[installation_id]
    tilt = metadata_id["Tilt"]
    peakPower = metadata_id["Watt Peak"]
    azimuth = metadata_id["Orientation"]
    latitude = metadata_id["Latitude"]
    longitude = metadata_id["Longitude"]
    
    ## Source data
    source_pvgis = data_slicer(pvgis, source_range)
    source_CEDA = data_slicer(CEDA, source_range)


    data = pd.merge(source_pvgis, source_CEDA, left_index=True, right_index=True)
    data = pd.merge(data, is_day, right_index=True, left_index=True)

    data = [data] # Put it in a list to work with featurisation object
    data = Featurisation(data)
    data.data = data.cyclic_features()
    data.data = data.add_shift('P')
    data.data = data.cyclic_angle('wind_direction_10m')
    data.data = data.PoA(latitude, longitude, tilt, azimuth)
    data.data = data.remove_outliers(tolerance=50)


    source_dataset = data.data[0]
   
    min = source_dataset.min(axis=0).to_dict()
    max = source_dataset.max(axis=0).to_dict()

    target_CEDA = data_slicer(CEDA, target_range)
    target_power = data_slicer(power, target_range)
    eval_CEDA = data_slicer(CEDA, eval_range)
    eval_power = data_slicer(power, eval_range)

    #TARGET DATASET
    data = pd.merge(target_power, target_CEDA, left_index=True, right_index=True, how='inner')
    data = pd.merge(data, is_day, right_index=True, left_index=True)
    data = [data] # Put it in a list to work with featurisation object
    data = Featurisation(data)
    data.data = data.cyclic_features()
    data.data = data.add_shift('P')
    data.data = data.cyclic_angle('wind_direction_10m')
    data.data = data.PoA(latitude, longitude, tilt, azimuth)
    data.data = data.remove_outliers(tolerance=50)

    target_dataset = data.data[0]

    ## EVALUATION DATASET
    data = pd.merge(eval_power, eval_CEDA, left_index=True, right_index=True, how='inner')
    data = pd.merge(data, is_day, right_index=True, left_index=True)
    data = [data] # Put it in a list to work with featurisation object
    data = Featurisation(data)
    data.data = data.cyclic_features()
    data.data = data.add_shift('P')
    data.data = data.cyclic_angle('wind_direction_10m')
    data.data = data.PoA(latitude, longitude, tilt, azimuth)
    data.data = data.remove_outliers(tolerance=50)

    eval_dataset = data.data[0]

    features = list(source_dataset.columns)
    features = ['P', 'temperature_1_5m', 'relative_humidity_1_5m', 'diffuse_surface_SW_flux', 'direct_surface_SW_flux', 'downward_surface_SW_flux', 'PoA', 'P_24h_shift', "is_day"]

    
    accuracy = forecast_maker(source_dataset, target_dataset, features, eval_dataset, trial, optimizer_name, lr_source, lr_target,
                               n_layers_source, n_layers_target, n_nodes_source, n_nodes_target, batch_size_source, batch_size_target,dropout, [min,max])

    return accuracy


if __name__ == "__main__":

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "No_feature_select"  # Unique identifier of the study.
    storage_name = "sqlite:///HP_source.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

