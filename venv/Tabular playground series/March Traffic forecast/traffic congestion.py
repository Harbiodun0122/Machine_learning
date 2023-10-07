"""You are to forecast twelve-hours of traffic flow in a major US metropolitan area. Time, space and directional features give you
the chance to model interactions across a network of roadways """

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# loading the dataset
dataset = pd.read_csv('train.csv', parse_dates=['time'])
test_dataset = pd.read_csv('test.csv', parse_dates=['time'])

# DATA EXPLORATION
print('Dataset head: \n', dataset.head())
print('Testing Dataset head: \n', test_dataset.head())
print('Dataset shape: \n', dataset.shape)
print('Testing Dataset shape: \n', test_dataset.shape)
print('Dataset info: \n', dataset.info())
print('Testing Dataset info: \n', test_dataset.info())
print('Dataset description: \n', dataset.describe)
print('Testing Dataset description: \n', test_dataset.describe)
print('Dataset null columns: \n', dataset.isnull().sum())
print('Testing Dataset null columns: \n', test_dataset.isnull().sum())
print('Dataset direction counter: \n', Counter(dataset['direction']))
print('Testing Dataset direction counter : \n', Counter(test_dataset['direction']))

# FEATURE EXTRACTION
# splitting the dataset into dependent and independent variable
X = dataset.drop(['row_id', 'time', 'congestion'], axis=1)
X_test = test_dataset.drop(['row_id', 'time'], axis=1)
y = dataset['congestion']


# getting day name, month name and the time out of the datatime series
def time_series(data):
    day_name, month_name, time = [], [], []
    # test_day_name, test_month_name, test_time = [], [], []
    for i in data:
        day_name.append(i.day_name())
        month_name.append(i.month_name())
        normal_time = i.time()
        # changing time to string so that it can be sliced and changed to an integer
        normal_time = str(normal_time)
        normal_time = normal_time.split(':')
        normal_time = ''.join(normal_time[:-1])
        # converting strings back to integer
        normal_time = int(normal_time)
        time.append(normal_time)
    return day_name, month_name, time


dataset_change = time_series(dataset['time'])
test_dataset_change = time_series(test_dataset['time'])

# adding time, day and month to X dataframe
X['time'] = dataset_change[2]
X['Day'] = dataset_change[0]
X['Month'] = dataset_change[1]

X_test['time'] = test_dataset_change[2]
X_test['Day'] = test_dataset_change[0]
X_test['Month'] = test_dataset_change[1]

print('X: \n', X)
print('X_test: \n', X_test)


# DATA VISUALISATION
def draw_bar_chat(parameter):
    plt.bar(parameter.keys(), parameter.values())
    plt.title = f'GRAPH OF PARAMETER AGAINST OCCURENCE'
    plt.xlabel = 'PARAMETERS'
    plt.ylabel = 'OCCURENCE'
    plt.show()


bar_chat_parameters = [Counter(dataset['direction']), Counter(X['Day']), Counter(X['Month'])]

# for parameter in bar_chat_parameters:
#     draw_bar_chat(parameter)

def congestion_compare(X, Y):
    plt.figure()
    plt.scatter(X, Y)
    plt.title = f'GRAPH OF DAY OF WEEK AGAINST CONGESTION'
    plt.xlabel = 'DAY OF WEEK'
    plt.ylabel = 'CONGESTION'
    plt.legend('The graph', loc='upper right')
    plt.show()


parameters = [X['Day'], X['Month'], X['direction'], X['x'], X['y']]
# for parameter in parameters:
#     day_of_the_week_congestion(parameter, y)

# DATA PREPROCESSING
pre_process = ColumnTransformer(
    transformers=[('encoders', OneHotEncoder(drop='first', handle_unknown='ignore'), ['direction', 'Day', 'Month'])],
    remainder='passthrough')

# Instantiate the model

grid_search = XGBRegressor(colsample_bynode=1, colsample_bytree=1, enable_categorical=False, colsample_bylevel=1,
                           gamma=0, max_depth=10, min_child_weight=1, n_estimators=100,
                           n_jobs=-1, random_state=42, reg_alpha=1,
                           reg_lambda=0, scale_pos_weight=0.5, subsample=1, tree_method='auto',
                           verbosity=2)

# Define the pipeline
model_pipeline = Pipeline(steps=[('pre_processing', pre_process), ('linear_regression', grid_search)])
print('model_pipeline: ', model_pipeline.get_params(deep=False)['steps'])
print('X: \n', X)

# Fit the pipeline with the training dataset
model_pipeline.fit(X, y)
print('X: \n', X)
# Display the model score and the prediction of the model
print(f'score: {model_pipeline.score(X, y)}')
pred = model_pipeline.predict(X)
print('pred: ', pred)
print('y: ', y)
score = mean_absolute_error(y, pred)
print('score: ', score)

prediction = model_pipeline.predict(X_test)
print('prediction: ', pd.DataFrame(prediction))

# Turning to csv file to be submitted on Kaggle
output = pd.DataFrame({'row_id':test_dataset['row_id'], 'congestion':np.round(prediction)})
# output.to_csv('Kaggle_Traffic_Forecast.csv', index=False)