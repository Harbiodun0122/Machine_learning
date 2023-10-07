'''few of the uses of naive bayes
 face recognition
 weather prediction
 medical diagnosis
 news classification
advantages of naive bayes classifier
::  very simple and easy to implement
::  needs less training data
::  handles both continous and discrete data
::  highly scalable with number of predictors and data points
::  as it is fast, it can be used in real time predictions
:: not sensitive to irrelevant issues'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns#; sns.set()
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer # tokenizing each word in the article
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

data = fetch_20newsgroups()
#print(data.target_names)

categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
              'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
              'talk.politics.misc', 'talk.religion.misc'] # also the same thing like data.target_names

# training the data on these categories
train = fetch_20newsgroups(subset='train', categories=data.target_names)
print('len of train', len(train.data))

# testing the data for these categories
test = fetch_20newsgroups(subset='test', categories=data.target_names)

# printing training data
# print(train.data)

# creating a model based on Multinomial naive bayes, we are throwing it into pipeline because what we are going to search for is a string
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# training the model with the train data
model.fit(train.data, train.target)

# creating labels for the test data
labels = model.predict(test.data)
#print(labels)

# plotting the confusion matrix
mat = confusion_matrix(test.target, labels)
acc = accuracy_score(test.target, labels)
# xticklabels, yticklabels show the names instead of just numbers
sns.heatmap(mat.T, square=True, annot=True, fmt='d', xticklabels=train.target_names, yticklabels=train.target_names)
# plotting heatmap of confusion matrix
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title(f'Accuracy: {acc}', fontsize=10)
# plt.show()

# predicting category on the new data based on trained model
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]
print(predict_category('what the fuck!'))