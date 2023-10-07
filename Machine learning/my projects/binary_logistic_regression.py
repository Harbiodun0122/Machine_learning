'''To let a student know if he\she will be will be admitted into a certain school or not based on their past records'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# loading the csv file
book = pd.read_csv(r'C:\Users\USER\Desktop\Machine Learning Full datasets1\Logistic Regression in R\binary.csv')
#print(book.describe())

# data visualisation
sns.heatmap(book.corr())
# plt.show()

# splitting it into x and y
X = book.iloc[:, 1:].values
y = book.iloc[:,0].values

# splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# instantiate the model
logreg = LogisticRegression(random_state=0)

# training the model
logreg.fit(X_train, y_train)

# predicting the outcome of the model
prediction = logreg.predict(X_test)
print('prediction', prediction)

# accuracy of the model
acc = logreg.score(X_test, y_test)
print('model accuracy: ', acc)

# confusion matrix
cm = confusion_matrix(y_test, prediction)
sns.heatmap(cm, annot=True, square=True, fmt='.1f')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Binary decision tree classifier', fontsize=10)
# plt.show()

##### deploying to production #####
pred = logreg.predict([[800,4,1]])
print(pred)
##### deploying to production #####