import pyarrow.parquet as pq
import pandas as pd

total_df = pd.DataFrame()
j=0
parquet_file = pq.ParquetFile('UK/30min.parquet')
for i in parquet_file.iter_batches(batch_size=1e6):
    if j %100 == 0:
        print('One step further.')
    df = i.to_pandas()
    total_df = pd.concat([total_df, df])
    j += 1

print(total_df)
total_df.to_pickle('UK/ProdUK.pkl')