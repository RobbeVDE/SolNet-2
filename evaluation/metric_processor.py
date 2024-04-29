import pandas as pd
import numpy as np
def metric_processor(accuracy, timer, i, j):
    n_sites = 4
    n_models = 12
    n_months = 13
    #Save mean metrics to put in table in report
    my_index = pd.MultiIndex.from_product([range(n_sites), range(n_months)], names=[u'one', u'two'])
    my_time_columns = pd.MultiIndex.from_product([range(n_models), ['Inference Time', 'Training Time']], names=[u'one', u'two'])
    try:
        avg_rmse = pd.read_pickle("Evaluation/avg_rmse.pkl")
    except:
        avg_rmse= pd.DataFrame(index=range(n_sites), columns=range(n_models))

    try:
        avg_times = pd.read_pickle("Evaluation/avg_times.pkl")
    except:
        avg_times =  pd.DataFrame(index=range(n_sites), columns=my_time_columns)

    try:
        rmse = pd.read_pickle("Evaluation/rmse.pkl")
    except:
        rmse= pd.DataFrame(index=my_index, columns=range(n_models))


    try:
        times = pd.read_pickle("Evaluation/times.pkl")
    except:
        times =  pd.DataFrame(index=my_index, columns=my_time_columns)

    

    avg_acc =  np.sqrt((np.sum(np.square(accuracy)))/n_months)
    avg_rmse.loc[j,i] = avg_acc
    rmse.loc[(j,slice(None)), i] = accuracy

    for key, value in timer.items():
        avg_times.loc[j,(i,key) ] = np.nanmean(value)
        times.loc[(j, slice(None)), (i, key)] = value
    
    print(avg_rmse)
    print(avg_times)
    print(rmse)
    print(times)

    avg_rmse.to_pickle("Evaluation/avg_rmse.pkl")
    avg_times.to_pickle("Evaluation/avg_times.pkl")
    rmse.to_pickle("Evaluation/rmse.pkl")
    times.to_pickle("Evaluation/times.pkl")