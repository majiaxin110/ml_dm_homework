import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from ass_A.AssA import FSFDP

agg_data: np.ndarray = np.loadtxt("../data/Aggregation.txt", delimiter=',')
# print(agg_data)
print(agg_data.size)
print(agg_data.shape)
print("Shape: (%d,%d)" % (agg_data.shape[0], agg_data.shape[1]))

plt.scatter(agg_data[:, 0], agg_data[:, 1])
plt.show()

# y_kmean = KMeans(n_clusters=5, random_state=9).fit_predict(X=agg_data)
# plt.scatter(agg_data[:, 0], agg_data[:, 1], c=y_kmean)
# plt.show()


def distance_calc(x: np.ndarray, y: np.ndarray):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


# cluster_cores = np.array([768, 340, 602, 47, 721, 191, 552, 498])
# FSFDP(agg_data, distance_func=distance_calc, cluster_cores=cluster_cores)

# 开始调参
# cluster_cores = np.array([768, 601, 743, 127, 320, 190, 555])
# FSFDP(agg_data, distance_func=distance_calc, t=0.015, gama_graph=True, cluster_cores=cluster_cores)

cluster_cores = np.array([768, 43, 602, 318, 723, 553, 191])
FSFDP(agg_data, distance_func=distance_calc, dc=1.84, gama_graph=True, cluster_cores=cluster_cores, isSave=True)
