import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

'''To predict wine quality '''
# loading the dataset
dataset = pd.read_excel(r'C:\Users\USER\Desktop\Machine Learning Full datasets1\Random Forest\R\winequality-red.xls')
# print(dataset.describe())
# print(dataset.head())

# to check if somewhere in the dataset is empty
# na = dataset.isna().sum()
# print(na)
imputer = SimpleImputer(missing_values=0, strategy='mean')
dataset = imputer.fit_transform(dataset)
dataset = pd.DataFrame(dataset)

# splitting to x and y
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1].values
# scale the 'free sulfur dioxide' and 'total sulfur dioxide' column
column_trans = ColumnTransformer(transformers=[('stsc', StandardScaler(), [5,6])], remainder='passthrough')
X = column_trans.fit_transform(X)

# splitting to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# creating the pipeline, loading an instance of random forest classifier, training and predicting
pipe = make_pipeline(column_trans, RandomForestClassifier(criterion='entropy', random_state=0))
pipe.fit(X_train, y_train)
# pipe.fit(X, y)
prediction = pipe.predict(X_test)
print('prediction: ', prediction)

# accuracy
accuracy = accuracy_score(y_test, prediction)
print('accuracy: ', accuracy)

#### To production ####
# the column 5 and 6 has been changed to column 1 and 2 because i used standard scaler
predict = pipe.predict([[10,47,11.6,0.58,0.66,2.2,0.074,1.0008,3.25,0.57,9]])
print('predict: ', predict)