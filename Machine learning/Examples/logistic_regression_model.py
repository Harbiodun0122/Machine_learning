from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

digits = load_digits()

# print('image data shape', digits.data.shape)
# print('label data shape', digits.target.shape)

# displaying some of the images and labels
def show_numbers():
    #plt.figure(figsize=(20,4))
    for index, (image, target) in enumerate(zip(digits.data[0:10], digits.target[0:10])):# digits.data = image, digits.target = target
        #print('image: ',image)
        #print('label:', target)
        plt.subplot(2,5, index + 1) # how our graph will look, we can tweak it the way we want and it's based on how we write our enumerate
        plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.copper)# to show, reshape it and give it color
        plt.title(f'Training:{target}', fontsize = 10)
    plt.show()
#show_numbers()

# dividing dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state= 2)

# print('X_train: ', X_train.shape)
# print('X_test: ',X_test.shape)
# print('y_train: ', y_train.shape)
# print('y_test: ', y_test.shape)

# making an instance of the model and training it
logisticregr = LogisticRegression()
logisticregr.fit(X_train, y_train)

# prediction
# predicting the output of the first element of the test set
#print(logisticregr.predict(X_test[0].reshape(1, -1)))# we reshape because we are only searching for one output
# predicting the output of the first 10 element of the test set
#print('predicted', logisticregr.predict(X_test[0:10]))
# predicting the entire dataset
prediction = logisticregr.predict(X_test)
print('prediction: \n', prediction)


# determinig the accuracy of the model, accuracy_score can't be used beacuse it does not support multi-class, multi-output, multi-label
score = logisticregr.score(X_test, y_test)
print('score: ', score)

# representing the confusion matrix in a heat map
def ConfusionMatrix():
    cm = confusion_matrix(y_test, prediction)
    #print(cm)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot = True, fmt = '.2f', linewidth = .5, square=True, cmap= plt.cm.viridis)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = f'Accuracy score: {score}'
    plt.title(all_sample_title , size= 15)
    plt.show()
#ConfusionMatrix()

def show_actual_and_predicted():
    index = 0
    classifiedIndex = []
    for predict, actual in zip(prediction, y_test):
        if predict == actual:
            classifiedIndex.append(index) # write the index number of the predicted value
        index += 1
    #print('classifiedIndex: ',classifiedIndex)
    plt.figure(figsize=(20, 5))
    for plotIndex, right in enumerate(classifiedIndex[0:30]):
        plt.subplot(6, 5, plotIndex + 1) # how our graph will look, we can tweak it the way we want and it's based on how we write our enumerate
        plt.imshow(np.reshape(X_test[right], (8,8)), cmap=plt.cm.gray) # to show, reshape it and give it color
        plt.title(f'Predicted: {prediction[right]}, Actual: {y_test[right]}', fontsize = 10)# show our prediction and actual in the index number of right
    plt.show()
show_actual_and_predicted()

