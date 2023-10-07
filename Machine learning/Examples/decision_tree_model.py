import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

'''To know if a particular individual will repay the loan lent to him by the bank or not'''

# loading data file
balance_data = pd.read_csv('Decision_Tree.csv', sep=',', header=0) # the header parameter does not count
#  just to see what we are working with
#print('dataset length: ', len(balance_data))
#print('balance_data.shape:', balance_data.shape)
#print(balance_data.head())

# separating the target variable
X = balance_data.values[:, :4]
Y = balance_data.values[:, -1]

# splitting dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# function to perform training with entropy
# max_depth=3 means it will only have 3 layers, min_samples_leaf=5 means it will have at least 5 leaves at the end
clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

# prediction
pred = clf_entropy.predict(X_test) # in the real world, the x_test will be the information from the customer
#print('y_test: ', y_test[:10]) # the actual one we used test (first 10)
#print('y_pred: ', pred[:10]) # the predicted value (first 10)

# accuracy
#accuracy = accuracy_score(y_test, pred)
#print(f'Accuracy is {accuracy}')

############ real world application ########################
def predict(data, train=X_train, model=clf_entropy):
    pred = clf_entropy.predict(data)
    return pred
print(predict([[406,10187,166,3068],[205,10016,395,3044],[359,14202,427,3399],[471,14117,560,3240]]))
########### real world application #########################

