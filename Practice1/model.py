#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import models
from keras import layers
import numpy as np
from keras.utils.np_utils import to_categorical

# 导入MNIST数据集
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
print('train_shape {} {}'.format(train_data.shape,train_labels.shape))
print('test_shape {} {}'.format(test_data.shape,test_labels.shape))


# 数据预处理
x_train = train_data.reshape((60000, 28, 28, 1))
x_train = x_train.astype('float32')/255
x_test = test_data.reshape((10000, 28, 28, 1))
x_test = x_test.astype('float32')/255
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)
print(x_train.shape, y_train.shape)

# 定义模型
def model_conv():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    return model

model = model_conv()
print(model.summary())

# 导入增强后的训练集
x_train = np.load('augmented_images.npy')
y_train = np.load('augmented_labels.npy')

plt.subplot(321)
plt.imshow(x_train[0],cmap=plt.get_cmap('gray'))
plt.subplot(322)
plt.imshow(x_train[1],cmap=plt.get_cmap('gray'))
plt.subplot(323)
plt.imshow(x_train[2],cmap=plt.get_cmap('gray'))
plt.subplot(324)
plt.imshow(x_train[3],cmap=plt.get_cmap('gray'))
plt.subplot(325)
plt.imshow(x_train[4],cmap=plt.get_cmap('gray'))
plt.subplot(326)
plt.imshow(x_train[5],cmap=plt.get_cmap('gray'))
# show
plt.show()


# 训练模型
history = model.fit(x_train, y_train, epochs=5, batch_size=1024, validation_split=0.1)

# 计算准确度
loss, acc = model.evaluate(x_test, y_test)
print('loss {}, acc {}'.format(loss, acc))

# 保存模型
model.save("my_mnist_model.h5")