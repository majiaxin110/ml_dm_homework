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

print("file reading done")

# 看看有多少种不同的类
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
# cat_dict: dict = dict()
# current_index = 0
# for each_cat in all_cat:
#     cat_dict[each_cat] = current_index
#     current_index += 1
print("Cats: %d" % len(all_cat))
cat_value_list = list(all_cat.values())
cat_value_list.sort(reverse=True)
for each_key in all_cat:
    if all_cat[each_key] == 53460 or all_cat[each_key] == 25719 or all_cat[each_key] == 23942 or all_cat[each_key] == 15834:
        print(each_key)
print(cat_value_list)
#
# # ---- 开始形成三分类用户消费数据----
# three_data_result = np.zeros(shape=(1, 3), dtype=np.float)
# current_index = 0
# for each_user_key in users_dict:
#     if current_index % 2000 == 0:
#         print("当前三分类进度 %d" % current_index)
#     each_user_store_ids = users_dict[each_user_key]
#     c_user_data = [0, 0, 0]
#     for each_store_id in each_user_store_ids:
#         c_store = stores_dict[each_store_id]
#         c_categories = c_store["categories"]
#         for each_cate in c_categories:
#             if each_cate == "Shopping":
#                 c_user_data[0] += 1
#             elif each_cate == "Food":
#                 c_user_data[1] += 1
#             elif each_cate == "Beauty & Spas":
#                 c_user_data[2] += 1
#     c_user_data = np.array(c_user_data, dtype=float)
#     if c_user_data.max() > 0:
#         # 归一
#         c_user_data: np.ndarray = c_user_data / c_user_data.sum()
#         c_user_data = c_user_data.reshape((1, 3))
#         three_data_result = np.append(three_data_result, c_user_data, axis=0)
#     current_index += 1
# print(three_data_result.shape)
# np.save("three", three_data_result)
three_data_result = np.load("three.npy")
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(three_data_result[:2000, 0], three_data_result[:2000, 1], three_data_result[:2000, 2])
ax.view_init(elev=30, azim=20)
plt.show()

print("TSNEing...")
tsneResult = TSNE(n_components=2)
Y = tsneResult.fit_transform(three_data_result[:6000, :])
print(Y.shape)
plt.scatter(Y[:, 0], Y[:, 1])
plt.show()

# ---- ----- ---- ---- ---- ----
# df = pd.DataFrame.from_dict(dataj, orient="index")
# df = df.infer_objects()
# # oneData = dataj["--6MefnULPED_I942VcFNA"]
# # print(oneData)
# print(df.info())

# 开始形成每个用户的数组
# print("开始形成每个用户的数组")
# history_data = np.zeros(shape=(len(users_dict), len(all_cat)), dtype=np.float)
# food_count = np.zeros(len(users_dict))
# user_index = 0
# for key in users_dict:
#     history_list: list = users_dict[key]
#     if user_index % 3000 == 0:
#         print("当前进度 %d" % user_index)
#     for i in range(len(history_list)):
#         current_store_id: str = history_list[i]
#         current_store: dict = stores_dict[current_store_id]
#         current_categories: list = current_store["categories"]
#         for each_category in current_categories:
#             history_data[user_index, cat_dict[each_category]] += 0.50
#     user_index += 1
# print("Max %d Min %d" % (history_data.max(), history_data.min()))

# print("PCAing....")
# pca = PCA(n_components="mle")
# history_data = history_data[:1000, :100]
# print(history_data.shape)
# newData = pca.fit_transform(history_data)
# print(newData.shape)

# print("TSNEing...")
# tsneResult = TSNE(n_components=3)
# Y = tsneResult.fit_transform(history_data[:800, :])
# print(Y.shape)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2])
# plt.show()
#
#
# def distance_calc(x: np.ndarray, y: np.ndarray):
#     return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

# 看看有多少种不同的城市
# all_city: dict = {}
# for key in stores_dict:
#     each_rest: dict = stores_dict[key]
#     if 'city' in each_rest.keys():
#         each_city: str = each_rest['city']
#         if each_city in all_city:
#             all_city[each_city] = all_city[each_city] + 1
#         else:
#             all_city[each_city] = 1
#     else:
#         print("One Data Point doesn't have city")
# print("城市数目：%d" % len(all_city))
# cityRests = list(all_city.values())
# cityRests.sort(reverse=True)
# print(cityRests)
# # one_city = max(all_city, key=all_city.get)
# # print(one_city)
# # print(all_city[one_city])
# for each_city_key in all_city:
#     if all_city[each_city_key] == 25601 or all_city[each_city_key] == 16456 or all_city[each_city_key] == 16192:
#         print(each_city_key)
# 商铺能否聚类？
# store_info = np.zeros(shape=(1, 2))
# index = 0
# for key in stores_dict:
#     if index % 2000 == 0:
#         print("store 进度 %d" % index)
#     each_store: dict = stores_dict[key]
#     if "review_count" in each_store and "stars" in each_store:
#         store_info = np.append(store_info, [[each_store["review_count"], each_store["stars"]]], axis=0)
#     index += 1
#
# plt.scatter(store_info[:, 0], store_info[:, 1])
# plt.show()
# FSFDP(Y, distance_func=distance_calc)
# possible_users = 0
# only_one_city = 0
# two_city = 0
# for each_user_key in users_dict:
#     each_user_list = users_dict[each_user_key]
#     c_only_one = True
#     c_cities: set = set()
#     for each_store_id in each_user_list:
#         store = stores_dict[each_store_id]
#         if store["city"] == "Las Vegas" or store["city"] == "Phoenix" or store["city"] == "Toronto":
#             c_cities.add(store["city"])
#     if len(c_cities) > 0:
#         possible_users += 1
#     if len(c_cities) == 1:
#         only_one_city += 1
#     if len(c_cities) == 2:
#         two_city += 1
# print("possible %d only one %d two %d" % (possible_users, only_one_city, two_city))
# np.save("after_tsne.txt", Y)
