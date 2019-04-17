import os
import math
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================================
# -----------------生成图片路径和标签的List------------------------------------

train_dir = 'data/input_photos'

food = []
label_food = []
nonfood = []
label_nonfood = []


# 获取路径下所有的图片路径名，存放到
# 对应的列表中，同时贴上标签，存放到label列表中。
def get_files(file_dir, ratio=0.2):
    for file in os.listdir(file_dir + '/food'):
        food.append(file_dir + '/food' + '/' + file)
        label_food.append(0)
    for file in os.listdir(file_dir + '/nonfood'):
        nonfood.append(file_dir + '/nonfood' + '/' + file)
        label_nonfood.append(1)

    # 对生成的图片路径和标签List做打乱处理把cat和dog合起来组成一个list（img和lab）
    image_list = np.hstack((food, nonfood))
    label_list = np.hstack((label_food, label_nonfood))

    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 从打乱的temp中再取出list（img和lab）
    # image_list = list(temp[:, 0])
    # label_list = list(temp[:, 1])
    # label_list = [int(i) for i in label_list]
    # return image_list, label_list

    # 将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    # 将所得List分为两部分，一部分用来训练tra，一部分用来测试val
    # ratio是测试集的比例
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))  # 测试样本数
    n_train = n_sample - n_val  # 训练样本数

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    result_tra_images = np.empty(shape=(len(tra_labels), 64, 64, 3))
    for i in range(len(tra_images)):
        img = Image.open(tra_images[i])
        # img = img.convert("L")
        result_tra_images[i] = np.array(img)

    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    result_val_images = np.empty(shape=(len(val_images), 64, 64, 3))
    for i in range(len(val_images)):
        img = Image.open(val_images[i])
        # img = img.convert("L")
        result_val_images[i] = np.array(img)

    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return result_tra_images, tra_labels, result_val_images, val_labels

