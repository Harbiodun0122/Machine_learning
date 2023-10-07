import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score

################### FIRST EXAMPLE #########################
''' To know if a social network ad will be purchased or not '''
# loading the dataset
dataset = pd.read_csv(
    r'C:\Users\USER\Desktop\Machine Learning Full datasets1\ML Algorithms dataset\SocialNetworkAds.csv')

# splitting to x and y
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values

# initiating column transformer, and using the one hot encoder
# we must use remainder='passthrough' so that it won't encoder every column of X and only the first[0] column
column_trans = ColumnTransformer(
    transformers=[('ohe', OneHotEncoder(categories='auto', sparse=False, drop='first'), [0])],
    remainder='passthrough')
X = column_trans.fit_transform(X)

# dividing to test and split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# creating the pipeline
pipe = make_pipeline(column_trans, DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=3))

# training the model
trian = pipe.fit(X_train, y_train)

# prediction
pred = pipe.predict(X_test)
print(pred)

# predicting the accuracy with f1_score
accu = f1_score(y_test, pred)
print('f1 accuracy: ', accu)


############ Deploying to production ################
def predict(Gender, Age, Estimated):
    ''' Predict if a social network ad will be purchased or not '''
    # map the gender to the one hot encoded figures so it won't give us an error
    gender = {'Male': 1.0, 'Female': 0.0}
    preds = pipe.predict([[gender[Gender], Age, Estimated]])
    if preds == 1:
        return 'Yes'
    return 'No'


# print(predict('Male', 27, 58000))
############ Deploying to production ################
#################### FIRST EXAMPLE #########################


########################### SECOND EXAMPLE #######################################
# loading the excel dataset
golf = pd.read_excel(
    r'C:\Users\USER\Desktop\Machine Learning Full datasets1\Machine Learning Tutorial Part 1 _ 2\Decision Tree- golf.xlsx')

# # Transform the categorical variable using One Hot Encoder
ohe = OneHotEncoder(drop='first', sparse=False)
golf = pd.DataFrame(ohe.fit_transform(golf))

# splitting into features and target
X = golf.iloc[:, :-1].values
y = golf.iloc[:, -1].values

'''To be used when launched to production', it can't be used inside the function because the datas will transform after pipe.fit'''
outlook = ohe.categories_[0]
outlook_dict = {outlook[0]: [0, 0], outlook[1]: [1, 0], outlook[2]: [0, 1]}

temp = ohe.categories_[1]
temp_dict = {temp[0]: [0, 0], temp[1]: [1, 0], temp[2]: [0, 1]}

humidity = ohe.categories_[2]
humidity_dict = {humidity[0]: [0], humidity[1]: [1]}

windy = ohe.categories_[3]
windy_dict = {windy[0]: [0], windy[1]: [1]}
'''####################################### THE END ###########################################'''

# test and train data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# creating the pipeline, training and predicting our model
pipe = make_pipeline(ohe,
                     DecisionTreeClassifier(criterion='entropy', random_state=0, min_samples_leaf=1))

pipe.fit(X, y)

predictt = pipe.predict(X)
# print('predicted: ', predictt)
# print(y)

# trying to inverse transform from the ohe but i don't kinda get it
# for preds in predictt:
#     if preds == 1:
#         print('Yes')
#     else:
#         print('No')

# accuracy
accuracy = f1_score(y, predictt)


# print('f1 accuracy: ', accuracy)

######### To production ##########
def decision(array, pipe=pipe):
    x = []
    # loop through the array
    for conditions in array:
        # map the conditions to the value in the dictionary
        conditions[0] = outlook_dict[conditions[0]]
        x.extend(conditions[0])
        conditions[1] = temp_dict[conditions[1]]
        x.extend(conditions[1])
        conditions[2] = humidity_dict[conditions[2]]
        x.extend(conditions[2])
        conditions[3] = windy_dict[conditions[3]]
        x.extend(conditions[3])

    prediction = pipe.predict([x])
    for preds in prediction:
        if preds == 1:
            print('Yes')
        else:
            print('No')


decision([['Rainy', 'Cool', 'Normal', False]])
######### To production ##########
########################### SECOND EXAMPLE #######################################
