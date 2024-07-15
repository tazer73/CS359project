## Import packages
import pandas as pd
import glob

def extract():
    #For each file type: Create a temporary dataframe and concatenate to the 'data' dataframe.
    for csvfile in glob.glob('*.csv'):
        tmp_df = pd.read_csv(csvfile)
        data = pd.concat([tmp_df], ignore_index=True)

    for jsonfile in glob.glob('*.json'):
        tmp_df = pd.read_json(jsonfile,lines=True)
        data = pd.concat([data, tmp_df], ignore_index=True)

    for parquetfile in glob.glob('*.parquet'):
        tmp_df = pd.read_parquet(parquetfile)
        data = pd.concat([data,tmp_df], ignore_index=True)
    
    #Return combined data.
    return data

data = extract()
print('data shape:', data.shape)
print(data)
