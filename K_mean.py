# K_mean is example of Hard clustering, where every point belongs only to one cluster.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs  # Correct import

# Generate sample data
X, y = make_blobs(n_samples=500, n_features=2, centers=5, random_state=3)

print(X.shape)
np.unique(y)

#data normalisation & visualisation

def normalise(X):
  u = X.mean(axis=0)
  std = X.std(axis=0)
  return (X-u)/std

X = normalise(X)

plt.scatter(X[:,0], X[:,1])
plt.show()

#init the K-center for k cluster

k=5
colors = ['green','red', 'blue', 'yellow', 'orange']
n_features = 2

centroids = {}


for i in range(k):

  center =2*(2*np.random.random((n_features,))-1)
  print(center)\

  centroids[i] = {
      'center': center,
      'color': colors[i],
      'points': []
  }

print(centroids)

def distance(x1, x2):
  return np.sqrt(np.sum((x1-x2)**2))

def assign_points(X, centroids):
  m = X.shape[0]

  # each point will be assigned to exactly one of the clusters
  for i in range(m):
    cdist = []
    cx = X[i]

    #find out distance of pt from each centroid
    for kx in range(k):
      d = distance(centroids[kx]['center'], cx)
      cdist.append(d)

    clusterId= np.argmin(cdist)

    #assign the point to the list of points that current_cluster holds
    centroids[clusterId]['points'].append(cx)

assign_points(X, centroids)
centroids

#Step - 2b

def updateCluster(centroids):

  #Update every centroid by taking a mean of points assigned to cluster
  for kx in range(k):
    pts = np.array(centroids[kx]['points'])

    #if a cluster has non-zero points
    if pts.shape[0] >0:
      newCenter = pts.mean(axis=0)
      centroids[kx]['center'] = newCenter
      centroids[kx]['points'] = []  # Xlear the list for step 2a


#visualisation

def plotCluster(centroids):
  for kx in range(k):
    pts = np.array(centroids[kx]['points'])

    #plot the points
    if pts.shape[0] > 0:
      plt.scatter(pts[:,0], pts[:,1], color=centroids[kx]['color'])

    #plot the cluster center (centroid)
    uk = centroids[kx]['center']
    plt.scatter(uk[0], uk[1], color=centroids[kx]['color'], marker='x')


plotCluster(centroids)
