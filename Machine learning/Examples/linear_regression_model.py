import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

companies = pd.read_csv('1000_companies.csv')
# print(companies.describe())

# Data visualization
# building the correlation matrix, just to see the visual correlation in the data
a = sns.heatmap(companies.corr())
# plt.show()

# one hot encode the categorical variable
column_trans = ColumnTransformer(transformers=[('ohe', OneHotEncoder(sparse=False, drop='first'), [3])], remainder='passthrough')
companies = column_trans.fit_transform(companies)
companies = pd.DataFrame(companies)
print(companies)
# splitting to x and y
X = companies.iloc[:, :-1].values
y = companies.iloc[:, -1].values

# dividing to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
#print('X_train', X_train)
print('X_test', X_test[:10])
#print('y_train', y_train)
#print('y_test', y_test)

# creating our model, training and prediction
pipe = make_pipeline(column_trans, LinearRegression())

# traning the pipeline
pipe.fit(X_train, y_train)

# prediction
prediction = pipe.predict(X_test)
print('predict: ', prediction)

# accuracy
accuracy = pipe.score(X_test, y_test)
print('accuracy: ', accuracy)