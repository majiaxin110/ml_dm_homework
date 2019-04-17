# TensorFlow and tf.keras
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from .batcher import get_files
import json

train_images, train_label, test_images, test_label = get_files("data/input_photos", 0.3)
class_names = ['food', 'nonfood']

# 预处理
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_label[i]])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64, 64, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_label, epochs=25)

test_loss, test_acc = model.evaluate(test_images, test_label)

print('Test accuracy:', test_acc)
if test_acc > 0.81:
    formal_images_str = []
    image_names = []
    for file in os.listdir("D:\\temporary\\photos_64048_user_719"):
        formal_images_str.append("D:\\temporary\\photos_64048_user_719" + "\\" + file)
        image_names.append(file)

    print("Resizing...")

    formal_images = np.empty(shape=(len(formal_images_str), 64, 64, 3))
    for i in range(len(formal_images_str)):
        if i % 2000 == 0:
            print("Current resizing: % d" % i)
        img = Image.open(formal_images_str[i])
        img = img.resize((64, 64))
        # img = img.convert("L")
        formal_images[i] = np.array(img)

    print("Predicting...")
    predictions = model.predict(formal_images)
    name_and_class = {}
    for i in range(len(image_names)):
        if i % 2000 == 0:
            print("Current write: % d" % i)
        name_and_class[image_names[i]] = int(np.argmax(predictions[i]))

    jsObj = json.dumps(name_and_class)

    fileObject = open('classification.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()