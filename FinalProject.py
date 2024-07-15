## Import packages
import pandas as pd
import glob

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