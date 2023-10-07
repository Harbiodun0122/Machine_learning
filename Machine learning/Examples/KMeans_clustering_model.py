import matplotlib.pyplot as plt
import seaborn as sns; sns.set() # for plot styling
import numpy as np
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import  pairwise_distances_argmin, accuracy_score

# # cluster_std is the standard deviation of the clusters, or we can say the distance between the clusters
# X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=.50, random_state=0)
# # print('X\n', X)
# # print('y_true\n', y_true)
#
# # data visualisation
# plt.scatter(X[:, 0], X[:, 1], s= 20)
# plt.show()
#
# #assign 4 clusters
# kmeans = KMeans(n_clusters=4) #n_clusters is the number of clusters and the number of centroids to generate
#
# # train and predict
# kmeans.fit(X)
# y_kmeans = kmeans.predict(X)
# print('y_kmeans: ', y_kmeans)
#
# ############# I think this is just for visualisation #####################
# def find_clusters(X, n_clusters, rseed=2):
#     #1. randomly choose clusters
#     rng = np.random.RandomState(rseed)
#     i = rng.permutation(X.shape[0])[:n_clusters] # generate random 4 numbers between 0 and 299
#     #print('i: ', i)
#     centers = X[i] # so our centers will be X in the index position of 'i'
#     #print('centers: \n', centers)
#
#     while True:
#         #2a.  assign labels based on closest center
#         labels = pairwise_distances_argmin(X, centers)
#         #2b. find new centers from the mean points
#         new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
#         # 2c. check for convergence, this may have a little flaw
#         if np.all(centers == new_centers):
#             break
#         centers = new_centers
#
#     return centers, labels
#
# centers, labels = find_clusters(X, 4)
# #print('labels: ', labels)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap='Accent')
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=1.0)
# plt.show()
# ############# I think this is just for visualisation #####################


################# SECOND EXAMPLE (COLOUR COMPRESSION)###########################

# 1. ###########################################################
from sklearn.datasets import load_sample_image
import warnings; warnings.simplefilter('ignore')
from sklearn.cluster import MiniBatchKMeans

flower = load_sample_image('china.jpg')
#print('flower: ',flower)
ax = plt.axes(xticks = [], yticks = []) # x and y ticks so it won't draw unnecessary lines on the flower
ax.imshow(flower)
#plt.show()
print('flower_shape: ',flower.shape)

# reshape the data to [n_samples x n_features], and rescale the colors so that they lie between 0 and 1
data = flower / 255 #use 0 ..1 scale
data = data.reshape(427 * 640, 3)
print('data shape: ', data.shape)

# visualize these pixels in this color space, using a subset of 10,000 pixels for efficiency
def plot_pixel(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    # choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    # print('i',i)
    colors = colors[i]
    #print('colors: ', colors)
    R, G, B = data[i].T # transpose

    fig, ax = plt.subplots(1, 2, figsize =(16, 6))
    ax[0].scatter(R, G, color = colors, marker='.')
    ax[0].set(xlabel = 'Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20)
    plt.show()
# plot_pixel(data, title='Input color space: 16 million possible colors')

kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

# plot_pixel(data, colors=new_colors, title='Reduced color space: 16 colors')

flower_recolored = new_colors.reshape(flower.shape)
fig, ax = plt.subplots(1, 2, figsize =(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(flower)
ax[0].set_title('original image', size = 16)
ax[1].imshow(flower_recolored)
ax[1].set_title('16- color image', size = 16)
#plt.show()

def production(data, model=kmeans):
    img = data / np.max(data)# use 0 ..1 scale
    img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    model.fit(img) # training the model
    new_img = model.cluster_centers_[model.predict(img)] # prediction
    image_reshaped = new_img.reshape(data.shape) # change it to 3 dimensional picture
    fig, ax = plt.subplots(1, 2, figsize =(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow(data)
    ax[0].set_title('original image', size=20)
    ax[1].imshow(image_reshaped)
    ax[1].set_title('compressed image', size=20)
    plt.show()
image = load_sample_image('me.jpg')
production(image)



# # 2. #################################################
# china = load_sample_image('china.jpg')
# #print('china:', china)
# ac = plt.axes(xticks=[], yticks=[])
# ac.imshow(china)
# #plt.show()
#
# # returns the dimension of the array
# print('china.shape: ',china.shape)
#
# # reshape the data to [n_samples x n_features], and rescale the colors so that the lie between 0 and 1
# pic = china / 255
# pic = pic.reshape(427* 640, 3)
# print('pic.shape: ', pic.shape)
#
# # visualise these pixels in the color space, using a subset of 10,000 pixels for efficiency
# def plot_pixels(data, title, colors=None, N=10000):
#     if colors is None:
#         colors = pic
#     rng = np.random.RandomState(0)
#     i = rng.permutation(data.shape[0])[:N]
#     colors = colors[i]
#     R, G, B = pic[i].T
#
#     fig, ac = plt.subplots(1, 2, figsize=(16, 6))
#     ac[0].scatter(R, G, color=colors, marker='.')
#     ac[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))
#     ac[1].scatter(R, G, color=colors, marker='.')
#     ac[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))
#
#     fig.suptitle(title, size=20)
#     plt.show()
# #plot_pixels(pic, title='Input color space: 16 million possible colors')
#
# kmean = MiniBatchKMeans(16) # used to reduce number of possible colors to 16
# kmean.fit(pic)
# new_color = kmean.cluster_centers_[kmean.predict(pic)]
# #plot_pixels(pic, colors=new_color, title='Reduced color space: 16 colors')
#
# china_recolored = new_color.reshape(china.shape)
# fig, ac = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
# fig.subplots_adjust(wspace=0.05)
# ac[0].imshow(china)
# ac[0].set_title('Original image', size=16)
# ac[1].imshow(china_recolored)
# ac[1].set_title('16-color image', size=16)
# plt.show()