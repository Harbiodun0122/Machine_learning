'''few of the uses of svm
 face detection
 text and hypertext classification
 classification of images
 bioinformatics
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs # it's just used to create some data
from sklearn.svm import SVC
''' svc is used for small datasets, for large datasets
    consider using :class:`sklearn.svm.LinearSVC` or
    :class:`sklearn.linear_model.SGDClassifier` instead, possibly after a
    :class:`sklearn.kernel_approximation.Nystroem` transformer.'''

# we create over 40 separable points
# n_samples is the length of our features, centers is (giraffe or crocodile)
X, y = make_blobs(n_samples=40, centers=2, random_state=20)
print('X: ', X)
print('y: ', y)
# fit the model, don't regularize the illustration purposes
clf = SVC(C=1, kernel='linear') # kernel is linear so that the graph will be plotted straight and not round or sigmoid
clf.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap=plt.cm.Paired) # s == size
#plt.show()

############################### To display graph #############################################
# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim() # x upper and lower limit
ylim = ax.get_ylim() # y upper and lower limit
print('xlim', xlim)
print('ylim', ylim)

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30) # xlim[0](start) , xlim[1](stop),  30 = Number of samples to generate. Default is 50. Must be non-negative.
yy = np.linspace(ylim[0], ylim[1], 30) # ylim[0](start) , ylim[1](stop),  30 = Number of samples to generate. Default is 50. Must be non-negative.
print('xx', xx)
print('yy', yy)
YY, XX = np.meshgrid(yy, xx)
print('XX', XX)
print('YY', YY)
xy = np.vstack([XX.ravel(), YY.ravel()]).T # used ravel to it can be reshaped to (n_samples,)
print('xy', xy)
Z = clf.decision_function(xy).reshape(XX.shape)
print('Z', Z)
ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=1.0, linestyles = ['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=10, linewidths=20, facecolors='none')
plt.show()
############################### To display graph #############################################

############ when launched into production ##############
new = [[3,4], [5,6],[7,8],[1,7], [10,2]]# new is an array of arrays of bunch of data coming in to be predicted
pred = clf.predict(new)
print(pred)
############ when launched into production ##############
######## when launched into production ##############