''' To classify if a customer will come to purchase a good based on the time, discount and free delivery '''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# loading the dataset
dataset = pd.read_excel(r'C:\Users\USER\Desktop\Machine Learning Full datasets1\Naive Bayes\Naive_Bayes_Dataset.xlsx')
# print(dataset.head())

# splitting to x and y
X = dataset.iloc[:, :-1].values
y = pd.DataFrame(dataset.iloc[:, -1])

# changing categorical variables to nominal variables
column_transx = ColumnTransformer(transformers=[('ohe', OneHotEncoder(sparse=False, drop='first'), [0, 1, 2])],
                                  remainder='passthrough')
X = column_transx.fit_transform(X)

# not necessary to use column transformer on this one, i just decided to practice something
column_transy = ColumnTransformer(transformers=[('ohe', OneHotEncoder(sparse=False, drop='first'), [0])])
y = column_transy.fit_transform(y).ravel()

# mapping the words to their equivalent in OneHotEncoder
day = column_transx.named_transformers_['ohe'].categories_[0]
day_dict = {day[0]: [0, 0], day[1]: [1, 0], day[2]: [0, 1]}
discount = column_transx.named_transformers_['ohe'].categories_[1]
discount_dict = {discount[0]: [0], discount[1]: [1]}
free_delivery = column_transx.named_transformers_['ohe'].categories_[2]
free_delivery_dict = {free_delivery[0]: [0], free_delivery[1]: [1]}

"# train test and split"
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# creating the pipeline, training and predicting
pipe = make_pipeline(column_transx, CategoricalNB())
pipe.fit(X, y)

predict = pipe.predict(X)

prediction = column_transy.named_transformers_['ohe'].inverse_transform(predict.reshape(-1, 1)).ravel()
# print(f'prediction: {prediction}')
# print(f'predict: {predict}')
# print('y_train: ', y)

# accuracy
accuracy = pipe.score(X, y)
print('accuracy: ', accuracy)

########## TO PRODUCTION ##############

def production(array, pipe=pipe):
    x = []
    for params in array:
        params[0] = day_dict[params[0]]
        x.extend(params[0])
        params[1] = discount_dict[params[1]]
        x.extend(params[1])
        params[2] = free_delivery_dict[params[2]]
        x.extend(params[2])

    pred = pipe.predict([x])
    prediction = column_transy.named_transformers_['ohe'].inverse_transform(pred.reshape(-1, 1)).ravel()
    print(f'prediction: {prediction}')


production([['Weekend', 'No', 'No']])
######### TO PRODUCTION ##############
tion([['Weekend', 'No', 'No']])
######### TO PRODUCTION ##############
