import numpy as np
from ass_A.AssA import FSFDP

agg_data = np.load("after_tsne.txt.npy")


def distance_calc(x: np.ndarray, y: np.ndarray):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


cluster_cores = np.array([201, 522, 1387, 1994, 555, 620])

FSFDP(agg_data, distance_func=distance_calc,cluster_cores=cluster_cores)
