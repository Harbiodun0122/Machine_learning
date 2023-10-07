'''To classify if a sample is a cupcake or a muffin'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# loading our data
dataset = pd.read_csv(
    r'C:\Users\USER\Desktop\Machine Learning Full datasets1\Machine Learning Tutorial Part 1 _ 2\Cupcakes vs Muffins.csv')
print(dataset.head())

# splitting dataset to x and y
X = dataset.iloc[:, 1:].values
Y = pd.DataFrame(dataset.iloc[:, 0])

# one hot encode y
ohe = OneHotEncoder(sparse=False, drop='first')
'''i used ravel so that the shape of Y would be (n_samples,) and not (n_samples,n_targets), 
if not used it doesn't affect the outcome but it does throw an error'''
Y = ohe.fit_transform(Y).ravel()

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=0)

# determining the value of K
# K = sqrt(len(Y))
# print(K) #K = 2

# creating the model
# determine parameters
# def determine(parameter):
#     rf, acc = [], []
#     for i in parameter:
#         rf.append(i)
#         model = KNeighborsClassifier(n_neighbors=3, p=i, metric='minkowski')
#         model.fit(X, Y)
#         pred2 = model.predict(X)
#         accur  = accuracy_score(Y, pred2)
#         acc.append(accur)
#     for i in zip(rf, acc):
#         if i[1] == max(acc):
#             print(f'number:{i[0]}\t accuracy:{i[1]}')
# #
#     plt.figure(figsize=(10,4.5))
#     plt.plot(rf, acc, label='Nearest neigbours', color='Blue')
#     plt.xlabel('p')
#     plt.ylabel('accuracy')
#     plt.legend(loc='lower right')
#     plt.show()
# determine(list(range(1,150)))

model = KNeighborsClassifier(n_neighbors=3, p=1, metric='minkowski')

# training the model
model.fit(X_train, Y_train)

# prediction
preds = model.predict(X_test)

prediction = ohe.inverse_transform(preds.reshape(-1, 1))
print('prediction:', prediction)

# accuracy
acc = accuracy_score(Y_test, preds)
print('accuracy: ', acc)


###### To production ############
def production(array, ohe=ohe, model=model):
    pred = model.predict([array])
    prediction = ohe.inverse_transform(pred.reshape(-1, 1))
    print('prediction: ', prediction)


production([55, 28, 3, 7, 5, 2, 0, 0])
###### To production ############
