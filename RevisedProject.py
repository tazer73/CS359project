## Import packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, SelectPercentile
from xgboost import XGBRegressor , XGBClassifier

def extract():
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
    
    return combined

data = extract()

#Transform section
data = data.dropna()

#Load section
def loadcsv(data: pd.DataFrame) -> None:
    """This function loads the argument dataframe into a csv file

    Args:
        data (pd.DataFrame): extracted nd transformed dataframe
    """
    data.to_csv('dataSet.csv', index=False)

loadcsv(data)

#Reading data section (Load to dataframe)
data = pd.read_csv('dataSet.csv')

#Drop duplicate values
data = data.drop_duplicates()

#Identify if dataset has missing data. There is no missing data.
print("No. of missing data in dataset: " + str(data.isna().sum().sum()))

# Remove the Heartbleed rows
data = data[data.Label != 'Heartbleed']

#Create attack DF (y axis)
attack = data[['Label']]

#Delete 'Label' from original dataset
data.drop('Label',axis=1, inplace=True)

#Performing univariate analysis.
#Drop features with variane less than .05
#data = data.loc[:, data.var(axis=0) >= 0.05]

#Encoding data
attack['Label'] = attack['Label'].replace('BENIGN', 0, regex=True)
attack['Label'] = attack['Label'].replace('DoS Hulk', 1, regex=True)
attack['Label'] = attack['Label'].replace('DoS GoldenEye', 1, regex=True)
attack['Label'] = attack['Label'].replace('DoS Slowhttptest', 1, regex=True)

#split the data
features = data.to_numpy()
label = attack.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.25, random_state=6969)

print(y_train.shape)
print(y_test.shape)

y_train = y_train.flatten()
y_test = y_test.flatten()