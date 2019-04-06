import numpy as np
from matplotlib import pyplot as plt


def FSFDP(agg_data: np.ndarray, distance_func, dc=None, t=0.02, gama_graph: bool = False, isSave: bool=False, cluster_cores: np.ndarray = None):
    print("Data Size:%d" % agg_data.size)
    print("Data Shape:", agg_data.shape)
    data_n: int = agg_data.shape[0]

    # 先计算距离矩阵 会花去O(n^2)时间呢
    print("Calculating distance...")
    distance = np.empty((data_n, data_n), dtype=float)
    d_sequence: np.ndarray = np.empty(int(0.5 * data_n * (data_n - 1)))
    current_seq_index = 0
    for i in range(data_n):
        for j in range(data_n):
            distance[i, j] = distance[j, i] = distance_func(agg_data[i], agg_data[j])
            if i > j:
                d_sequence[current_seq_index] = distance[i, j]
                current_seq_index += 1
    # print(distance)
    if dc is None:
        # 确定截断距离参数dc
        d_sequence = np.sort(d_sequence)
        dc = d_sequence[int(np.around(t * d_sequence.size))]
        print("当前选取截断距离：%f" % dc)
    print("Calculating ρ...")
    # 开始计算ρ啦 复杂度O(n^2)呢
    rou: np.ndarray = np.zeros(data_n)
    for i in range(data_n):
        for j in range(data_n):
            if distance[i][j] < dc:
                rou[i] += 1

    # 生成一下关于ρ的降序排列下标序列
    q_of_rou = np.argsort(-rou)
    # 开始计算δ啦
    print("Calculating δ...")
    delta: np.ndarray = np.empty(data_n)
    relation: np.ndarray = np.empty(data_n, dtype=np.int)
    delta[int(q_of_rou[0])] = distance.max()  # 最高密度处的δ就是最大距离呢
    for i in range(1, data_n):  # 对于从第q[1]至q[n]
        delta[int(q_of_rou[i])] = distance.max()
        # 寻找比其密度大的点至其的最小距离
        for j in range(0, i):
            if distance[int(q_of_rou[j]), int(q_of_rou[i])] < delta[int(q_of_rou[i])]:
                delta[int(q_of_rou[i])] = distance[int(q_of_rou[j]), int(q_of_rou[i])]
                relation[int(q_of_rou[i])] = int(q_of_rou[j])  # 与邻居

    # δ也算好了，我们应该可以输出决策图了
    plt.scatter(rou, delta, alpha=0.5)
    for i in range(data_n):
        plt.text(rou[i], delta[i], str(i))
    plt.title("Decision Graph")
    plt.show()

    # 输出γ图以辅助选择聚类中心
    if gama_graph:
        print("ρ min max :%f %f" % (rou.min(), rou.max()))
        print("δ min max :%f %f" % (delta.min(), delta.max()))
        gama = rou * delta
        gama = -np.sort(-gama)
        n = np.arange(data_n)
        plt.scatter(n, gama)
        plt.title("γ = ρδ")
        plt.show()

    if cluster_cores is not None:
        color_cluster: np.ndarray = np.empty(data_n)
        for i in range(data_n):
            color_cluster[i] = -1
        for i in range(cluster_cores.size):
            color_cluster[int(cluster_cores[i])] = i
        for i in range(data_n):
            if color_cluster[int(q_of_rou[i])] == -1:
                color_cluster[int(q_of_rou[i])] = color_cluster[int(relation[int(q_of_rou[i])])]
        plt.scatter(agg_data[:, 0], agg_data[:, 1], c=color_cluster)
        # for i in range(cluster_cores.size):
        #     plt.scatter(agg_data[cluster_cores[i], 0], agg_data[cluster_cores[i], 1], c='', marker='o',
        #                 edgecolors='r', s=150)
        plt.show()
        if isSave:
            file_data = np.c_[agg_data, color_cluster]
            resultFile = open("result/task1.csv", "w")
            for i in range(data_n):
                resultFile.write("%f,%f,%d\n" % (file_data[i][0], file_data[i][1], file_data[i][2]))
            resultFile.close()
            print("Save done")
