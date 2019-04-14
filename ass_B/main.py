import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

from ass_A.AssA import FSFDP

three_data = np.load("three_cats_data.npy")
print("Data shape", end="")
print(three_data.shape)


# FSFDP(Sci.2004)
# def distance_calc(x: np.ndarray, y: np.ndarray):
#     return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)
#
#
# # FSFDP(three_data_result, t=0.015, distance_func=distance_calc, gama_graph=True)
# #
# start = time.clock()
# cores = np.array([4945, 2980, 485, 3882])
# fsfdp_classification = FSFDP(three_data_result, t=0.015, distance_func=distance_calc, gama_graph=True, cluster_cores=cores)
# elapsed = (time.clock() - start)
# print("Sci2014 time used ", elapsed)
#
# np.save("FSFDP/sci2014_class_result", fsfdp_classification)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(three_data_result[:, 0], three_data_result[:, 1], three_data_result[:, 2], c=fsfdp_classification)
# ax.view_init(elev=30, azim=20)
# plt.title("Sci2014 Result")
# plt.show()
#
# print("TSNEing")
# tsne = TSNE(n_components=2)
# tsne_result = tsne.fit_transform(three_data_result[:, :])
# print(tsne_result.shape)
# plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=fsfdp_classification)
# plt.title("Sci2014 After TSNE")
# plt.show()

# ---- KMeans ----
# start = time.clock()
# km_class_result = KMeans(n_clusters=5, random_state=9).fit_predict(three_data_result)
# elapsed = (time.clock() - start)
# print("KMeans time used ", elapsed)
# print(km_class_result.shape)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(three_data_result[:, 0], three_data_result[:, 1], three_data_result[:, 2], c=km_class_result)
# ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
# ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
# ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
# ax.view_init(elev=30, azim=20)
# plt.title("KMeans Result")
# plt.show()
# np.save("KMeans/kmeans_class_result", km_class_result)

# print("TSNEing")
# tsne = TSNE(n_components=2)
# tsne_result = tsne.fit_transform(three_data_result[:, :])
# print(tsne_result.shape)
# plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=km_class_result)
# plt.title("KMeans After TSNE")
# plt.show()

# ---- DBSCAN ----
# start = time.clock()
# db_classification_result = DBSCAN(eps=0.09).fit_predict(three_data_result)
# elapsed = (time.clock() - start)
# print("DBSCAN time consumed ", elapsed)
# np.save("DBSCAN/dbscan_class_result", db_classification_result)
#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(three_data_result[:, 0], three_data_result[:, 1], three_data_result[:, 2], c=db_classification_result)
# ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
# ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
# ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
# ax.view_init(elev=30, azim=20)
# plt.title("DBSCAN Result")
# plt.show()
#
# # print("TSNEing")
# # tsne = TSNE(n_components=2)
# # tsne_result = tsne.fit_transform(three_data_result)
# # print(tsne_result.shape)
# # plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=db_classification_result)
# # plt.title("DBSCAN After TSNE")
# # plt.show()
#
# print("类别数 %d" % (db_classification_result.max() + 1))

# ---- Hierarchical ----
start = time.clock()
hierarchical_result = AgglomerativeClustering(n_clusters=4, linkage="complete").fit_predict(three_data)
elapsed = (time.clock() - start)
print("Hierarchical time consumed ", elapsed)
np.save("Hierarchical/hierarchical_complete_class_result", hierarchical_result)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(three_data[:, 0], three_data[:, 1], three_data[:, 2], c=hierarchical_result)
ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
ax.view_init(elev=30, azim=20)
plt.title("Hierarchical Result")
plt.show()

# print("TSNEing")
# tsne = TSNE(n_components=2)
# tsne_result = tsne.fit_transform(three_data)
# print(tsne_result.shape)
# plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=hierarchical_result)
# plt.title("Hierarchical After TSNE")
# plt.show()

# ---- 谱聚类 ----
# start = time.clock()
# spectral_result = SpectralClustering(n_clusters=4).fit_predict(three_data)
# elapsed = (time.clock() - start)
# print("Spectral time consumed ", elapsed)
# np.save("Spectral/spectral_class_result", spectral_result)
#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(three_data[:, 0], three_data[:, 1], three_data[:, 2], c=spectral_result)
# ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
# ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
# ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
# ax.view_init(elev=30, azim=20)
# plt.title("Spectral Result")
# plt.show()

# ---- EM-GMM ----
# start = time.clock()
# gmm = GaussianMixture(n_components=4).fit(three_data)
# gmm_result = gmm.predict(three_data)
# elapsed = (time.clock() - start)
# print("EM-GMM time consumed ", elapsed)
# np.save("EM_GMM/em_gmm_class_result", gmm_result)
#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(three_data[:, 0], three_data[:, 1], three_data[:, 2], c=gmm_result)
# ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
# ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
# ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
# ax.view_init(elev=30, azim=20)
# plt.title("EM-GMM Result")
# plt.show()
