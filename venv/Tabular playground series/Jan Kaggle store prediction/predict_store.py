import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# loading the training and testing data
data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# Data Exploration

print(f'Training data info: {data.info()}\n')
print(f'Training data description: {data.describe()}\n')
print(f'Testing data info: {test_data.info()}\n')
print(f'Null columns: {data.isnull().sum()}\n')

# to see if the data is balanced by counting the number of occurences of each data in the columns
from collections import Counter
print(f'Country counter: {Counter(data["country"])}\n')
print(f'Store counter: {Counter(data["store"])}\n')
print(f'Product counter: {Counter(data["product"])}\n')



# Separating the data into independent and dependent variables

train_x = data.drop(columns=['num_sold'])
train_y = data['num_sold']



# Data Visualisation

def show_fig(column, x_label, y_label):
    plt.figure(num=column.capitalize())
    plt.bar(train_x[column], train_y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


show_fig('country', 'country', 'num_sold')
show_fig('store', 'store', 'num_sold')
show_fig('product', 'product', 'num_sold')
# plt.show()

'''From the data visualisation above, it has been seen that
Norway is the country with highest sales
KaggleRama is the store with highest sales
KaggleHat is the most sold product
'''



# Data preprocessing
pre_process = ColumnTransformer(transformers=[('encoders', OneHotEncoder(drop='first', handle_unknown='ignore'), ['country', 'store', 'product'])],
                                remainder='drop')
print(pre_process)


# Instantiate the model
grid_search = RandomForestRegressor(n_estimators=100, criterion="squared_error", random_state=0, n_jobs=-1)

# Define the pipeline
model_pipeline = Pipeline(steps=[('pre_processing', pre_process), ('decision_tree', grid_search)])
print('model_pipeline: ',model_pipeline.get_params(deep=False)['steps'])

# Fit the pipeline with the training dataset
model_pipeline.fit(train_x, train_y)

# Display the model score and the prediction of the model
print(f'score: {model_pipeline.score(train_x, train_y)}')
prediction = model_pipeline.predict(test_data)
print('prediction: ', pd.DataFrame(prediction))


# Turning to csv file to be submitted on Kaggle
output = pd.DataFrame({'row_id':test_data['row_id'], 'num_sold':np.round(prediction)})
output.to_csv('Kaggle_Tabular_Playground.csv', index=False)