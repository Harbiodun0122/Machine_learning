''' The boston dataset'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

# loading the dataset
boston = load_boston()
# print(boston.DESCR)
# feature_name = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

# splitting to data and target
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target
# print(X)
# print(X.describe())

# data visualisation
plt.figure(figsize=(10,4.5))
# plt.plot(X['AGE'][-10:], y[-10:], marker='>', label='age')
# plt.title('Boston house price')
# plt.legend()
# plt.show()

# splitting to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# model
# determining parameters
# def determine(parameter):
#     rf, acc = [], []
#     for i in parameter:
#         rf.append(i)
#         rforest = RandomForestRegressor(random_state=0, max_leaf_nodes=41, min_samples_split=17)
#         rforest.fit(X_train, y_train)
#         pred2 = rforest.predict(X_test)
#         accur  = rforest.score(X_test, y_test)
#         acc.append(accur)
#     for i in zip(rf, acc):
#         if i[1] == max(acc):
#             print(f'number:{i[0]}\t accuracy:{i[1]}')

#     plt.figure(figsize=(10,4.5))
#     plt.plot(rf, acc, label='Random forest', color='Blue')
#     plt.xlabel('max_leaf_nodes')
#     plt.ylabel('accuracy')
#     plt.legend(loc='lower right')
#     plt.show()
# determine(list(range(2,150)))

rforest = RandomForestRegressor(random_state=0, max_leaf_nodes=41, min_samples_split=17)

# fitting the model
rforest.fit(X, y)

# prediction
prediction = rforest.predict(X)
# print(f'prediction: {prediction}')
# print('y_test: ', y)

# accuracy
score = rforest.score(X, y)
print(f'accuracy: {score}')