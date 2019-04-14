import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=2000, n_features=3,
                  centers=[[-1, -1, -1], [0, 0, 0], [1, 1, 2], [2, 3, 1]], cluster_std=[0.3, 0.4, 0.2, 0.1],
                  random_state=None)

fig = plt.figure()
ax = Axes3D(fig)
# ax.scatter(X[:, 0], X[:, 1], X[:, 2])
# ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
# ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
# ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
# plt.show()

y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(X)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred)
ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
plt.show()
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.show()
#
# # 评估分数
# print(metrics.calinski_harabaz_score(X, y_pred))
