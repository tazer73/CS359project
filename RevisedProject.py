## Import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, Ridge
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, RFE
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression
from sklearn.ensemble import AdaBoostClassifier, RandomForestRegressor, VotingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_score, recall_score, accuracy_score
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

#Identify shape of the dataset.
print("Dataset shape: " + str(data.shape))
#Cleanup Column names
data.columns = data.columns.str.strip()
data.columns = data.columns.str.replace(" " , "_")
data.columns = data.columns.str.replace(".1" , "")

#Identify if dataset has missing data. There is no missing data.
print("No. of missing data in dataset: " + str(data.isna().sum().sum()))

# Remove the Heartbleed rows
data = data[data.Label != 'Heartbleed']

#Create attack DF (y axis)
attack = data[['Label']]

#Delete 'Label' from original dataset
data.drop('Label',axis=1, inplace=True)

#Performing univariate analysis.
#Drop features with variance less than .05 (removing 0 variance)
data = data.loc[:, data.var(axis=0) >= 0.05]

#Encoding data
attack['Label'] = attack['Label'].replace('BENIGN', 0, regex=True)
attack['Label'] = attack['Label'].replace('DoS Hulk', 1, regex=True)
attack['Label'] = attack['Label'].replace('DoS GoldenEye', 1, regex=True)
attack['Label'] = attack['Label'].replace('DoS Slowhttptest', 1, regex=True)

#split the data
features = data.to_numpy()
label = attack.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.25, random_state=6969)

y_train = y_train.flatten()
y_test = y_test.flatten()

#standardize the data
scaler = StandardScaler()
X_train_standard = scaler.fit_transform(X_train)
X_test_standard = scaler.fit(X_test)

#Describe the data to ensure correct standardization
#print(pd.DataFrame(X_train_standard).describe())

#Feature selection
xgbC = XGBClassifier()
xgbC.fit(X_train_standard,y_train)
selection = SelectFromModel(xgbC,threshold=.01, prefit=True)
X_train_selected = selection.transform(X_train)
X_test_selected = selection.transform(X_test)
#End of feature selection

#XGB, Perceptron, AdaBoost, Logistic Regression, Majority Voting

#XGBC MODEL
xgbC.fit(X_train_selected , y_train)
X_train_selected_pred = xgbC.predict(X_train_selected)
X_test_selected_pred = xgbC.predict(X_test_selected)

train_acc_perc = accuracy_score(y_train, X_train_selected_pred)
train_f1score_perc = f1_score(y_train, X_train_selected_pred)
train_precision_perc = precision_score(y_train, X_train_selected_pred)
train_recall_perc = recall_score(y_train, X_train_selected_pred)

print('-'*20)
print('\nTrain XGBC Model')
print('Accuracy: {:.3f}'.format(train_acc_perc))
print('Precision: {:.3f}'.format(train_precision_perc))
print('Recall: {:.3f}'.format(train_recall_perc))
print('F1-score: {:.3f}'.format(train_f1score_perc))

test_acc_perc = accuracy_score(y_test, X_test_selected_pred)
test_f1score_perc = f1_score(y_test, X_test_selected_pred)
test_precision_perc = precision_score(y_test, X_test_selected_pred)
test_recall_perc = recall_score(y_test, X_test_selected_pred)

print('\nTest XGBC Model')
print('Accuracy: {:.3f}'.format(test_acc_perc))
print('Precision: {:.3f}'.format(test_precision_perc))
print('Recall: {:.3f}'.format(test_recall_perc))
print('F1-score: {:.3f}'.format(test_f1score_perc))

#PERCEPTRON MODEL
perceptron = Perceptron(eta0=0.001, random_state=1)
perceptron.fit(X_train_selected,y_train)
X_train_selected_pred = perceptron.predict(X_train_selected)
X_test_selected_pred = perceptron.predict(X_test_selected)

train_acc_perc = accuracy_score(y_train, X_train_selected_pred)
train_f1score_perc = f1_score(y_train, X_train_selected_pred)
train_precision_perc = precision_score(y_train, X_train_selected_pred)
train_recall_perc = recall_score(y_train, X_train_selected_pred)

print('-'*20)
print('\nTrain PERCEPTRON Model')
print('Accuracy: {:.3f}'.format(train_acc_perc))
print('Precision: {:.3f}'.format(train_precision_perc))
print('Recall: {:.3f}'.format(train_recall_perc))
print('F1-score: {:.3f}'.format(train_f1score_perc))

test_acc_perc = accuracy_score(y_test, X_test_selected_pred)
test_f1score_perc = f1_score(y_test, X_test_selected_pred)
test_precision_perc = precision_score(y_test, X_test_selected_pred)
test_recall_perc = recall_score(y_test, X_test_selected_pred)

print('\nTest PERCEPTRON Model')
print('Accuracy: {:.3f}'.format(test_acc_perc))
print('Precision: {:.3f}'.format(test_precision_perc))
print('Recall: {:.3f}'.format(test_recall_perc))
print('F1-score: {:.3f}'.format(test_f1score_perc))

#ADABOOST MODEL
adaboost_perc = AdaBoostClassifier(estimator=Perceptron(eta0=0.001, random_state=1), n_estimators=100, learning_rate=0.001, random_state=1, algorithm='SAMME').fit(X_train_selected, y_train)
adaboost_perc.fit(X_train_selected,y_train)
X_train_selected_pred = adaboost_perc.predict(X_train_selected)
X_test_selected_pred = adaboost_perc.predict(X_test_selected)

train_acc_perc = accuracy_score(y_train, X_train_selected_pred)
train_f1score_perc = f1_score(y_train, X_train_selected_pred)
train_precision_perc = precision_score(y_train, X_train_selected_pred)
train_recall_perc = recall_score(y_train, X_train_selected_pred)

print('-'*20)
print('\nTrain ADABOOST Model')
print('Accuracy: {:.3f}'.format(train_acc_perc))
print('Precision: {:.3f}'.format(train_precision_perc))
print('Recall: {:.3f}'.format(train_recall_perc))
print('F1-score: {:.3f}'.format(train_f1score_perc))

test_acc_perc = accuracy_score(y_test, X_test_selected_pred)
test_f1score_perc = f1_score(y_test, X_test_selected_pred)
test_precision_perc = precision_score(y_test, X_test_selected_pred)
test_recall_perc = recall_score(y_test, X_test_selected_pred)

print('\nTest ADABOOST Model')
print('Accuracy: {:.3f}'.format(test_acc_perc))
print('Precision: {:.3f}'.format(test_precision_perc))
print('Recall: {:.3f}'.format(test_recall_perc))
print('F1-score: {:.3f}'.format(test_f1score_perc))

#LOGISTIC REGRESSION MODEL
lr = LogisticRegression().fit(X_train_selected, y_train)
lr.fit(X_train_selected,y_train)
X_train_selected_pred = lr.predict(X_train_selected)
X_test_selected_pred = lr.predict(X_test_selected)

train_acc_perc = accuracy_score(y_train, X_train_selected_pred)
train_f1score_perc = f1_score(y_train, X_train_selected_pred)
train_precision_perc = precision_score(y_train, X_train_selected_pred)
train_recall_perc = recall_score(y_train, X_train_selected_pred)

print('-'*20)
print('\nTrain LOGISTIC REGRESSION MODEL')
print('Accuracy: {:.3f}'.format(train_acc_perc))
print('Precision: {:.3f}'.format(train_precision_perc))
print('Recall: {:.3f}'.format(train_recall_perc))
print('F1-score: {:.3f}'.format(train_f1score_perc))

test_acc_perc = accuracy_score(y_test, X_test_selected_pred)
test_f1score_perc = f1_score(y_test, X_test_selected_pred)
test_precision_perc = precision_score(y_test, X_test_selected_pred)
test_recall_perc = recall_score(y_test, X_test_selected_pred)

print('\nTest LOGISTIC REGRESSION MODEL')
print('Accuracy: {:.3f}'.format(test_acc_perc))
print('Precision: {:.3f}'.format(test_precision_perc))
print('Recall: {:.3f}'.format(test_recall_perc))
print('F1-score: {:.3f}'.format(test_f1score_perc))

#MAJORITYVOTING MODEL
hard_majorityvote = VotingClassifier(estimators=[('perceptron', perceptron),('adaboost_perc', adaboost_perc),('logistic regression', lr)],voting='hard')
hard_majorityvote.fit(X_train_selected,y_train)
X_train_selected_pred = hard_majorityvote.predict(X_train_selected)
X_test_selected_pred = hard_majorityvote.predict(X_test_selected)

train_acc_perc = accuracy_score(y_train, X_train_selected_pred)
train_f1score_perc = f1_score(y_train, X_train_selected_pred)
train_precision_perc = precision_score(y_train, X_train_selected_pred)
train_recall_perc = recall_score(y_train, X_train_selected_pred)

print('-'*20)
print('\nTrain MAJORITYVOTING MODEL')
print('Accuracy: {:.3f}'.format(train_acc_perc))
print('Precision: {:.3f}'.format(train_precision_perc))
print('Recall: {:.3f}'.format(train_recall_perc))
print('F1-score: {:.3f}'.format(train_f1score_perc))

test_acc_perc = accuracy_score(y_test, X_test_selected_pred)
test_f1score_perc = f1_score(y_test, X_test_selected_pred)
test_precision_perc = precision_score(y_test, X_test_selected_pred)
test_recall_perc = recall_score(y_test, X_test_selected_pred)

print('\nTest MAJORITYVOTING MODEL')
print('Accuracy: {:.3f}'.format(test_acc_perc))
print('Precision: {:.3f}'.format(test_precision_perc))
print('Recall: {:.3f}'.format(test_recall_perc))
print('F1-score: {:.3f}'.format(test_f1score_perc))