#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import models
from keras import layers
import numpy as np
from keras.utils.np_utils import to_categorical
import os
import random
import tensorflow as tf
from augment import Augument_MNIST

# 固定随机数种子保证实验可重复
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

set_global_determinism(seed=42)

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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model

model = model_conv()
print(model.summary())

# 数据增强(模拟摄像头输入), 数据扩增 20 倍，训练集，键盘输入‘y’保存到本地
# Augument_MNIST(N_times=20)

# 如果已经做了数据增强，则导入增强后的训练集
if os.path.exists('augmented_images.npy') and os.path.exists('augmented_labels.npy'):
    x_train = np.load('augmented_images.npy')
    y_train = np.load('augmented_labels.npy')

    
# 训练模型
history = model.fit(x_train, y_train, epochs=15, batch_size=1024, validation_split=0.1)

# 计算测试准确度
loss, acc = model.evaluate(x_test, y_test)
print('loss {}, acc {}'.format(loss, acc))

# 保存模型
model.save("my_mnist_model.h5")

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()
plt.savefig("")

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()