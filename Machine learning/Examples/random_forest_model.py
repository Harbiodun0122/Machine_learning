from sklearn.datasets import load_iris
from sklearn.ensemble import  RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

''' iris flower analysis '''

# setting the random seed
np.random.seed(0)

# creating an object called iris with the data
iris = load_iris()
#print('iris: ', iris)
#print('iris: ', iris.target_names)

# creating a dataframe with the four variables
df = pd.DataFrame(iris.data, columns=iris.feature_names) # we have to convert it to pandas so that we can be able to view it on a table

# adding a new column (the type of the flower)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names) # mapping the target to the target name
print("df['species']:", df['species'])

# creating test and train data
# (0, 1,) means numbers will be generated between 0 and 1, <=.75 means if is_train is less than .75(75% is the training size) then it's true else, false
df['is_train'] = np.random.uniform(0, 1, len(df)) <=.75 # this is the parameter the man used to choose his test and train
print(df)
# creating dataframes with the test rows and training rows
train, test = df[df['is_train'] == True], df[df['is_train'] == False]
print('train :\n', train)
# print('test :\n', test)

# show the number of observations for the test and training dataframes
#print('Number of observations in the training data: ', len(train))
#print('Number of observations in the testing data: ', len(test))

# create a list of the feature column's names
features = df.columns[:4]
print('features: ', features)# Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# converting each species name into digits
y = pd.factorize(train['species'])[0] # it is also the same as iris.target of 'train'. This method(pd.factorise) is useful for obtaining a numeric
# representation of an  array when all that matters is identifying distinct values. maybe also the same as the map function
print('y: ', y)

# creating random forest classifier
clf = RandomForestClassifier(n_jobs=2, random_state=0) # n_jobs is to prioritize

# training the classifier
clf.fit(train[features], y) # like our normal x_train and y_train

# applying the trained classifier to the test
predict = clf.predict(test[features])
print('predict: ', predict)
            # or
# mapping names for the plants for each predicted plants for each predicted plants
preds = iris.target_names[predict]
print('preds: ', preds)

#################### This may not be necessary #####################
# viewing the predicted probabilities of the first 10 observations
clf.predict_proba(test[features])
# print('clf.predict_proba(test[features]): ', clf.predict_proba(test[features]))
################### This may not be necessary #######################

# viewing the actual species for the first five observations
#print(test['species'].head()) # or print(test.species.head())

# creating confusion matrix
matrix = pd.crosstab(test['species'], preds, rownames=['Actual species'], colnames=['Predicted species']) # just same as the confusion matrix
# print(matrix)
###### a nice chart ##########
cm = confusion_matrix(test.species, preds)
#print(cm)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot = True, fmt = '.3f', linewidths = .5, square=True, cmap= 'Blues_r', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.ylabel('Actual species')
plt.xlabel('Predicted species')
# plt.show()
###### a nice chart ##########

##### deploying to production ####
pred = iris.target_names[clf.predict([[5.0, 3.6, 1.4,56],[3.0, 1.6, 4.4, 6.0]])]
print('Specie: ', pred)
##### deploying to production ####

###### my own way of doing things ######
acc = accuracy_score(test.species, preds)
# print('accuracy: ', acc)
###### my own way of doing things ######