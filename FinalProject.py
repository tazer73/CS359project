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
from xgboost import XGBRegressor # type: ignore

#Extract section
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

print("WITHOUD DROIPPING: " + str(data.shape))
data = data.drop_duplicates()
print(" DROIPPING: " + str(data.shape))


#7. EXPLORATORY ANALYSIS

#Identify shape of the dataset.
print("Original dataset shape: " + str(data.shape))
#Cleanup Column names
data.columns = data.columns.str.strip()
data.columns = data.columns.str.replace(" " , "_")
data.columns = data.columns.str.replace(".1" , "")

#Identify if dataset has missing data. There is no missing data.
print("No. of missing data in dataset: " + str(data.isna().sum().sum()))

# Remove the Heartbleed rows
data = data[data.Label != 'Heartbleed']
print("Dataset shape without 'Heartbleed' elements: " + str(data.shape))
#Create attack DF (y axis)
attack = data[['Label']]
print("New 'attack' dataset shape: " + str(attack.shape))
#Delete 'Label' from original dataset
data.drop('Label',axis=1, inplace=True)
print("Base dataset without 'Label' column: " + str(data.shape))


#Performing univariate analysis.
#Drop features with variane less than .05
#print(data.describe())
data = data.loc[:, data.var(axis=0) >= 0.05]
#print(data.head)
#print(attack.head)

#data.hist()
#(pd.DataFrame(X.AT).corrwith(X.V))

#Encoding data
attack['Label'] = attack['Label'].replace('BENIGN', 1, regex=True)
attack['Label'] = attack['Label'].replace('DoS Hulk', 2, regex=True)
attack['Label'] = attack['Label'].replace('DoS GoldenEye', 3, regex=True)
attack['Label'] = attack['Label'].replace('DoS Slowhttptest', 4, regex=True)

features = data.to_numpy()
label = attack.to_numpy()

#split the data
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.25, random_state=6969)

#standardize the data
scaler = StandardScaler()
X_train_standard = scaler.fit_transform(X_train)
X_test_standard = scaler.fit_transform(X_test)

lr = LinearRegression()

#WITHOUT FEATURE SELECTION
rf_regressor = RandomForestRegressor(n_estimators=70)
# train the model
rf_regressor.fit(X_train_standard, y_train.ravel())
# make predictions
pred = rf_regressor.predict(X_test_standard)
# compute r2-score and mse
r2 = r2_score(y_test, pred)
print("r2 score WITHOUT: {:.3f}".format(r2))
# compute mse
mse = mean_squared_error(y_test, pred)
print("mse: {:.3f}".format(mse))

# SELECT PERCENTILE FEATURES
r_selection = SelectPercentile(r_regression, percentile=30)
# model fitting and feature selection
X_train_selected = r_selection.fit_transform(X_train_standard, y_train.ravel())
X_test_selected = r_selection.transform(X_test_standard)
# fit the regressor using the selected features
rf_regressor.fit(X_train_selected, y_train.ravel())
print('X_train_selected shape:', X_train_selected.shape)
print('X_test_selected shape:', X_test_selected.shape)
# make predictions
pred = rf_regressor.predict(X_test_selected)
# compute r2-score and mse
r2 = r2_score(y_test, pred)
print("r2 score PERCENTILE: {:.3f}".format(r2))
# compute mse
mse = mean_squared_error(y_test, pred)
print("mse: {:.3f}".format(mse))


# SELECT K-BEST
r_selection = SelectKBest(r_regression, k=20)
# use the training set to train the model, then transform the training set
X_train_selected = r_selection.fit_transform(X_train_standard, y_train.ravel())
# transform the test set
X_test_selected = r_selection.transform(X_test_standard)
print('X_train_selected shape:', X_train_selected.shape)
print('X_test_selected shape:', X_test_selected.shape)
# fit the random forest regressor using the selected features
rf_regressor.fit(X_train_selected, y_train.ravel())
# make predictions
pred = rf_regressor.predict(X_test_selected)
## different runs may lead to different results
# compute r2-score and mse
r2 = r2_score(y_test, pred)
print("r2 score K-BEST: {:.3f}".format(r2))
# compute mse
mse = mean_squared_error(y_test, pred)
print("mse: {:.3f}".format(mse))

# XGBOOST
xgb_regressor = XGBRegressor(n_estimators=100, max_depth=10)
# fit the model using the entire dataset
xgb_regressor.fit(X_train_standard, y_train.ravel())
# display feature importance
xgb_regressor.feature_importances_
# store feature importance in a variable
feature_importance = xgb_regressor.feature_importances_
# preform feature selecting based on a 0.01 feature importance threshold
X_train_selected = X_train_standard[:, feature_importance > 0.01]
X_test_selected = X_test_standard[:, feature_importance > 0.01]
print('X_train_selected shape:', X_train_selected.shape)
print('X_test_selected shape:', X_test_selected.shape)
# model fitting using the training dataset with selected features
xgb_regressor.fit(X_train_selected, y_train.ravel())
# make predictions
pred = xgb_regressor.predict(X_test_selected)
# regularization has a great improvement boost
# compute r2-score and mse
r2 = r2_score(y_test, pred)
print("r2 score XGB: {:.3f}".format(r2))

# compute mse
mse = mean_squared_error(y_test, pred)
print("mse: {:.3f}".format(mse))