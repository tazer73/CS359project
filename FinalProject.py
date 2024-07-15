## Import packages

import pandas as pd
import glob

def extract():
    """This function combines all the csv, json, and parquet files into a dataframe
    
    Args:
        None: The function reads all csv and json finles in the working directory
        
    Returns: 
        data (pd.dataframe): All data sources combined in a single dataframe
    """

    #Create an empty dataframe to hold all the data.
    #Column names remain as they were int the original files.
    data = pd.DataFrame(columns=[' Destination Port', ' Flow Duration', ' Total Fwd Packets',
       ' Total Backward Packets', 'Total Length of Fwd Packets',
       ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
       ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
       ' Fwd Packet Length Std', 'Bwd Packet Length Max',
       ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
       ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s',
       ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
       'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
       ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
       ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
       ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
       ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
       ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean',
       ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count',
       ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
       ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count',
       ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size',
       ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
       ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
       ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
       'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes',
       ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
       ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
       ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max',
       ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min',
       ' Label'])
    
    #For each file type: Create a temporary dataframe and concatenate to the 'data' dataframe.
    for csvfile in glob.glob('*.csv'):
        tmp_df = pd.read_csv(csvfile)
        data = pd.concat([data,tmp_df], ignore_index=True)

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
print('column names:', data.columns)

# Load the CSV files
csv0 = pd.read_csv("ids_0.csv")
csv1 = pd.read_csv("ids_1.csv")
csv2 = pd.read_csv("ids_2.csv")
combined = pd.concat([ csv0 , csv1 , csv2], ignore_index=True)

# Load the JSON files
json3 = pd.read_json("ids_3.json", lines=True)
json4 = pd.read_json("ids_4.json", lines=True)
json7 = pd.read_json("ids_7.json", lines=True)
json9 = pd.read_json("ids_9.json", lines=True)
json10 = pd.read_json("ids_10.json", lines=True)
combined = pd.concat([combined , json3 , json4 , json7 , json9 , json10], ignore_index=True)

# Load the Parquet files
parquet5 = pd.read_parquet("ids_5.parquet")
parquet6 = pd.read_parquet("ids_6.parquet")
parquet8 = pd.read_parquet("ids_8.parquet")
parquet11 = pd.read_parquet("ids_11.parquet")
combined = pd.concat([combined , parquet5 , parquet6 , parquet8 , parquet11], ignore_index=True)

# Display the combined DataFrame
print(combined)
