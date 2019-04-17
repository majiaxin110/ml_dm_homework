import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ass_A.AssA import FSFDP
import json

stores = open("../data/business_163665.json", "r").read()
stores_dict = json.loads(stores)

users = open("../data/user_business_223699.json", "r").read()
users_dict = json.loads(users)

print("reading file done")

# 所有商店的各种类别
all_cat: dict = dict()
for key in stores_dict:
    each_rest: dict = stores_dict[key]
    if 'categories' in each_rest.keys():
        cat_list: list = each_rest['categories']
        for each_cat in cat_list:
            if each_cat in all_cat:
                all_cat[each_cat] += 1
            else:
                all_cat[each_cat] = 0
    else:
        print("One Data Point doesn't have category")

print("总类别数 %d" % len(all_cat))

# 数量前10的类别
top_cats = sorted(all_cat.items(), key=lambda item: item[1], reverse=True)
print(top_cats[:10])
# ---- 开始形成三类店铺用户消费数据----
three_data_result: np.ndarray
if not os.path.exists("three_cats_data.npy"):
    three_data_result = np.zeros(shape=(0, 3), dtype=np.float)
    unique_data_dict = dict()
    all_history = []
    unique_line_dict = dict()
    current_index = 0
    scanned_user_amount = 0
    current_three_index = 0
    for each_user_key in users_dict:
        if current_index % 4000 == 0:
            print("当前三分类进度 %d" % current_index)
        each_user_store_ids = users_dict[each_user_key]
        c_user_data = [0, 0, 0]
        for each_store_id in each_user_store_ids:
            c_store = stores_dict[each_store_id]
            c_categories = set(c_store["categories"])
            if "Restaurants" in c_categories or "Food" in c_categories:
                c_user_data[0] += 1
            if "Shopping" in c_categories:
                c_user_data[1] += 1
            if "Beauty & Spas" in c_categories:
                c_user_data[2] += 1
        data_hash = c_user_data[0]*10000 + c_user_data[1]*100 + c_user_data[2]
        all_history.append(max(c_user_data))
        # 转换为ndarray
        c_user_data = np.array(c_user_data, dtype=float)
        if c_user_data.max() > 0:
            scanned_user_amount += 1
            if data_hash in unique_data_dict:
                current_index += 1
                equal_list: list = unique_data_dict[data_hash]
                equal_list.append(each_user_key)
                equal_list = unique_line_dict[current_three_index]
                equal_list.append(each_user_key)
                continue
            else:
                unique_data_dict[data_hash] = [each_user_key]
            # 归一
            c_user_data: np.ndarray = c_user_data / c_user_data.sum()
            c_user_data = c_user_data.reshape((1, 3))
            three_data_result = np.append(three_data_result, c_user_data, axis=0)
            current_three_index += 1
            unique_line_dict[current_three_index] = [each_user_key]
        current_index += 1
    print(three_data_result.shape)
    print("覆盖的用户数 %d" % scanned_user_amount)
    print("unique line %d" % len(unique_line_dict))
    np.save("three_cats_data", three_data_result)
    jsObj = json.dumps(unique_data_dict)
    fileObject = open('datahash_user_id.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()
    jsObj = json.dumps(unique_line_dict)
    fileObject = open('each_line_id.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()
else:
    three_data_result = np.load("three_cats_data.npy")
    print("Data shape", end="")
    print(three_data_result.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(three_data_result[:, 0], three_data_result[:, 1], three_data_result[:, 2])
    ax.set_zlabel('L', fontdict={'size': 10, 'color': 'red'})
    ax.set_ylabel('M', fontdict={'size': 10, 'color': 'red'})
    ax.set_xlabel('N', fontdict={'size': 10, 'color': 'red'})
    ax.view_init(elev=30, azim=20)
    plt.show()

    print("TSNEing...")
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(three_data_result[:6000, :])
    print(tsne_result.shape)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
    plt.title("After TSNE")
    plt.show()
