'''To diagnose if a person has an eye problem or not'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

dataset = pd.read_csv(
    r'C:\Users\USER\Desktop\Machine Learning Full datasets1\Machine Learning Tutorial Part 1 _ 2\Part 2\data (1) (dataset for logistic).csv')
# print(dataset.isna().sum())

X = dataset.iloc[:, 2:]
Y = dataset.iloc[:, 1].values
# print(X.describe().columns)

# replacing missing values with the mean of the column
imputer = SimpleImputer(missing_values=0, strategy='mean')
X = imputer.fit_transform(X)

# using the standard scaler after using simple imputer
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X))

# Data visualisation, my visualisation is not that good
data = dataset.corr()
# sns.heatmap(data, annot=True, square=True)
# plt.show()

# one hot encode the categorical variable, its not important in this case, but i'm using it because of f1_score
ohe = OneHotEncoder(categories='auto', sparse=False, drop='first')
Y = ohe.fit_transform(Y.reshape(-1, 1)).ravel()

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=0)

''' to determine the number of maximum iteration of our model, from our graph it was shown that 9 is least number of max_iter that will converge quickly and won't give an error'''
# def max_iter(num):
#     c, accu = [], []
#     for i in num:
#         c.append(i)
#         model = LogisticRegression(solver='liblinear', random_state=0, max_iter=i)
#         model.fit(X, Y)
#         prediction = model.predict(X)
#         acc = accuracy_score(Y, prediction)
#         accu.append(acc)
#     for i in zip(c, accu):
#         if i[1] == max(accu):
#             print(f'number: {i[0]} \t accuracy: {acc}')
#     plt.figure(figsize=(10,4))
#     plt.plot(c, accu)
#     plt.xlabel('max_iter')
#     plt.ylabel('accuracy')
#     plt.show()
# max_iter(list(range(1,10)))

# instantiate the model
pipe = make_pipeline(imputer, scaler,  LogisticRegression(solver='liblinear', random_state=0, max_iter=9))

# training the model
pipe.fit(X_train, Y_train)

# prediction
prediction = pipe.predict(X_test)
y = ohe.inverse_transform(
    prediction.reshape(-1, 1)).ravel()  # convert it back to the original one so it can be readable by a normal person
print('Prediction', y)

# accuracy
acc = accuracy_score(Y_test, prediction)
print('accuracy: ', acc)