import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_csv('SMSSpamCollection.csv')
# print(dataset.head())

X = dataset.iloc[:, -1].values
y = dataset.iloc[:, 0].values

count_v = CountVectorizer()
X = count_v.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=100)

model = MultinomialNB()
model.fit(X, y)

prediction = model.predict(X)
print(prediction)

# accuracy
accuracy = accuracy_score(y, prediction)
print(accuracy)

# print(classification_report(y, prediction))

def prediction(s,count_v=count_v,  model=model):
    count_v.fit_transform(s)
    prediction = model.predict([[s]])
    print(prediction)
prediction('Rofl. Its true to its name')
