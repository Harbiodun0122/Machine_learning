import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler # if diff btw features is large, it is used to change features to range from -1, to 1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

'''one of the simplest supervised machine learning algorithm mostly used for classification
    USE CASE: DIABETES PREDICTION'''

dataset = pd.read_csv('diabetes.csv')

# values of columns like 'Glucose','BloodPressure','SkinThickness','BMI','Insulin' cannot be accepted as zeroes because it will affect the outcome,
# we can replace such values with the mean of the respective column
zero_not_accepted = ['Glucose','BloodPressure','SkinThickness','BMI','Insulin']
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)# replace 0 with NaN
    mean = int(dataset[column].mean(skipna=True)) #skipna means it will skip all the NaN
    dataset[column] = dataset[column].replace(np.NaN, mean) # replace NaN with mean
#print(dataset.head())

# split the dataset
X = dataset.iloc[: , 0:8]
y = dataset.iloc[: , 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print('X_train 1: \n', X_train)
# print('X_test1: \n', X_test)

'''Rule of thumb: 
    Any algorithm that computes distance or assumes normality, scale our features'''
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print('X_train 2: \n', X_train)
#print('X_test2: \n', X_test)

# training the model
'''n_neigbours refers to 'K', p(whether the patient is diabetic or not) is the power parameter 
to define the metric used which euclidean in our case.
To determine our K we find the sqrt of the len of our data
>>> from math import sqrt
>>> sqrt(len(y_test)
>>> 12.409673645990857
'K' cannot be an even number, so 1 is subtracted from it.'''
knn = KNeighborsClassifier(n_neighbors=11, p=None, metric='euclidean')
knn.fit(X_train, y_train)

# prediction
prediction = knn.predict(X_test)
#print('prediction: ', prediction)

# accuracy of the model, using one is just okay!
print('accuracy: ', accuracy_score(y_test, prediction))
print('cm: ', confusion_matrix(y_test, prediction))
print('f1_score:' ,f1_score(y_test, prediction))

######### just did it on my own ###########################
def graphic_confusion_matrix():
    '''just for fine visualisation of the confusion matrix'''
    cm = confusion_matrix(y_test, prediction)
    plt.figure(figsize=(2,2)) # may not be important
    # linewidths may not be important, cmap is for the color, square is to make each cell look like a square
    sns.heatmap(cm, annot = True, fmt = '.2f', linewidths = .5, square=True, cmap= 'RdPu')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = f'accuracy: {accuracy_score(y_test, prediction)}'
    plt.title(all_sample_title , size= 15)
    plt.show()
# graphic_confusion_matrix()
######### just did it on my own ###########################