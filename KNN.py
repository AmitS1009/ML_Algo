import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
plt.style.use("seaborn")


X, y = make_blobs(n_samples=2000, n_features=2, centers=3, random_state=42)
print(X.shape, y.shape)


m = X.shape[0]  # or len(y)

xt = np.array([-0.5, 1])

for i in range(m):
  if y[i] == 0:
    plt.scatter(X[i,0], X[i, 1], c="r", label = "red")
  elif y[i]==1:
    plt.scatter(X[i,0], X[i, 1], c="b", label = "green")
  else:
    plt.scatter(X[i,0], X[i, 1], c="g", label = "blue")

plt.scatter(xt[0], xt[1], color='orange', marker = '*')
plt.show()


def dist(p, q):
  return np.sqrt(np.sum((p-q)**2))

def knn(X, y, xt, k=10):
  m = X.shape[0]
  dlist = []

  for i in range(m):

    d = dist(X[i], xt)
    dlist.append((d, y[i]))

  dlist = sorted(dlist)
  dlist = np.array(dlist[:5])
  labels = dlist[:,1]

  print(labels)
  print(dlist[:k])

  labels , cnts = np.unique(labels, return_counts = True )
  print(labels, cnts)
  idx = np.argmax(cnts)
  pred = labels[idx]

  return int(pred)

xt = np.array([1, 1])

knn(X,y, xt)
