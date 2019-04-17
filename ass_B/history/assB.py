import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ass_A.AssA import FSFDP
import json

photo_class_file = open("../../data/photo_classification.json", "r").read()
photo_class = json.loads(photo_class_file)

user_photos_file = open("../../data/user_photo_719.json", "r").read()
user_photo = json.loads(user_photos_file)

print("file reading done")

result_data = []
result_index = 0
not_classified = 0
for user_key in user_photo:
    photo_id_list = user_photo[user_key]
    if len(photo_id_list) >= 800:
        continue
    result_data.append(len(photo_id_list))
    food_counter = 0
    for each_photo_id in photo_id_list:
        each_photo_name = each_photo_id + ".jpg"
        if each_photo_name not in photo_class:
            print(each_photo_id + " Not classified once")
            not_classified += 1
            continue
        if photo_class[each_photo_name] == 0:
            food_counter += 1
    result_data.append(food_counter)
    result_index += 1
result_data = np.array(result_data, dtype=np.float)
result_data = result_data.reshape((int(result_data.size / 2), 2))
# result_data[:, 0] = result_data[:, 0] / result_data[:, 0].max()
# result_data[:, 1] = result_data[:, 1] / result_data[:, 1].max()

print("Data transfer done")
print(result_data.max())
print("Not classified %d" % not_classified)
plt.scatter(result_data[:, 0], result_data[:, 1])
plt.xlabel("Picture")
plt.ylabel("Food")
plt.show()