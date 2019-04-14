import csv
import json
import time

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

from ass_A.AssA import FSFDP

three_data = np.load("three_cats_data.npy")
print("Three Data shape", end="")
print(three_data.shape)

sci2014_result = np.load("FSFDP/sci2014_class_result.npy")
kmeans_result = np.load("KMeans/kmeans_class_result.npy")
dbscan_result = np.load("DBSCAN/dbscan_class_result.npy")
em_gmm_result = np.load("EM_GMM/em_gmm_class_result.npy")
hierarchical_result = np.load("Hierarchical/hierarchical_average_class_result.npy")
hierarchical_c_result = np.load("Hierarchical/hierarchical_complete_class_result.npy")
spectral_result = np.load("Spectral/spectral_class_result.npy")

# Silhouette Coefficient
# print("sci2014 score ", metrics.silhouette_score(three_data, sci2014_result, metric="euclidean"))
# print("KMeans score ", metrics.silhouette_score(three_data, kmeans_result, metric="euclidean"))
# print("DBSCAN score ", metrics.silhouette_score(three_data, dbscan_result, metric="euclidean"))
# print("EM-GMM score ", metrics.silhouette_score(three_data, em_gmm_result, metric="euclidean"))
# print("hierarchical(Average) score ", metrics.silhouette_score(three_data, hierarchical_result, metric="euclidean"))
# print("hierarchical(Complete) score ", metrics.silhouette_score(three_data, hierarchical_c_result, metric="euclidean"))
# print("spectral score ", metrics.silhouette_score(three_data, spectral_result, metric="euclidean"))

# 组合得到最终数据
json_file = open("each_line_id.json", "r").read()
datahash_user_id: dict = json.loads(json_file)

with open("assB.csv", mode='w', newline='') as assBFinal:
    resultWriter = csv.writer(assBFinal, delimiter=',')
    for t_data_index in range(three_data.shape[0]):
        c_user_list = datahash_user_id[str(int(t_data_index + 1))]
        for each_user_id in c_user_list:
            resultWriter.writerow([each_user_id, int(sci2014_result[t_data_index]), int(kmeans_result[t_data_index]),
                                   int(dbscan_result[t_data_index]), int(hierarchical_result[t_data_index]),
                                   int(spectral_result[t_data_index]), int(em_gmm_result[t_data_index])])
    assBFinal.close()
print("done")
