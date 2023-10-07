'''The task is to classify 10 genomic bacteria species using data firn a genomic analysis technique that
has some data compression and data loss. In this technique, 10-mer snippets of DNA are sampled and analysed to give the histogram of
base count. In other words, the DNA  segment ATATGGCCTT becomes A2TG2C2. Use thiss lossy information in the train.csv to accurately
 predict bacteria species'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score


# LOADING THE DATASET
dataset = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')

# DATA ANALYTICS
print('dataset.describe: \n', dataset.describe())
print('dataset.info()): \n', dataset.info())
print('dataset.shape: \n', dataset.shape)
print('dataset.isnull(): \n', dataset.isnull().sum().sum())

# FEATURE SELECTION
# split into train and test
X_train = dataset.drop(['row_id', 'target'], axis=1)
y_train = dataset['target']
print('X_train: \n', X_train)
print('y_train: \n', y_train)

test_train = testdata.drop('row_id', axis=1)
print('test_train: ', test_train)

# DATA VISUALISATION
# check if the dataset is balanced
plt.pie(Counter(y_train).values(), labels=Counter(y_train).keys(), shadow=True, autopct='%1.2f%%', startangle=90)
# plt.show()

# DATA PREPROCESSING
# scaling the independent data
ct = ColumnTransformer([('standard_scaler', StandardScaler(), [i for i in range(0, 286)])])
X_train = ct.fit_transform(X_train)
test_train = ct.fit_transform(test_train)
print('Transformed X_train: \n', X_train)
print('Transformed X_train: \n', test_train)
# encoding the target categorical variable
ohe = LabelEncoder()
y_train = ohe.fit_transform(y_train)
print('Encoded y_train: \n', y_train)

# MODEL SELECTION
# creating and training the model
lr = ExtraTreesClassifier(random_state=0)
lr.fit(X_train, y_train)
# making predictions with the model
prediction = lr.predict(test_train)
print('prediction: ', ohe.inverse_transform(prediction))

# Turning to csv file to be submitted on Kaggle
output = pd.DataFrame({'row_id':testdata['row_id'], 'target':ohe.inverse_transform(prediction)})
output.to_csv('Feb_Kaggle_Tabular_Playground.csv', index=False)