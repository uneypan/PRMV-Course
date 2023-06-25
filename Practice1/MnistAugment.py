
from keras.datasets import mnist
import numpy as np
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

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


from keras.preprocessing.image import ImageDataGenerator


# 添加椒盐噪声函数
def salt_and_pepper_noise(image):
    noisy_image = np.copy(image)
    noise_mask = np.random.choice([0, 1, 2], size=image.shape[:2], p=[0.05, 0.05, 0.9])

    salt = noise_mask == 0
    pepper = noise_mask == 1

    noisy_image[salt] = 1
    noisy_image[pepper] = 0

    return noisy_image


# 创建图像数据生成器
datagen = ImageDataGenerator(
    rotation_range=10,      # 随机旋转角度范围
    width_shift_range=0.1,  # 随机水平平移范围
    height_shift_range=0.1, # 随机垂直平移范围
    shear_range=0.1,        # 随机剪切变换范围
    zoom_range=0.1,         # 随机缩放范围
    fill_mode='nearest',    # 填充模式
    preprocessing_function=salt_and_pepper_noise # 添加椒盐噪声
)

# 对训练集进行数据增强
datagen.fit(x_train)


# 生成增广后的数据
augmented_data_generator = datagen.flow(x_train, y_train, batch_size=60000, shuffle=False)

N = 10  # 增广次数

augmented_images = []
augmented_labels = []
for i in range(N):
    print("第{}次增广".format(i+1))
    augmented_data = next(augmented_data_generator)
    augmented_images.append(augmented_data[0])
    augmented_labels.append(augmented_data[1])

x_train = np.concatenate(augmented_images, axis=0)
y_train = np.concatenate(augmented_labels, axis=0)

# 保存增广后的数据
np.save('augmented_images.npy', x_train)
np.save('augmented_labels.npy', y_train)

plt.imshow(x_train[0])
plt.show()
