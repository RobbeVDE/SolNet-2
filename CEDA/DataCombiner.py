import pandas as pd
total_df = pd.DataFrame()
for i in range(1,12):
    file = f"CEDA/CEDA_data{i}.pickle"
    df = pd.read_pickle(file)
    total_df = pd.concat([total_df, df])

total_df.to_pickle("CEDA_data.pickle")