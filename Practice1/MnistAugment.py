
from keras.datasets import mnist
import numpy as np
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import cv2
from skimage.util import random_noise
from skimage.transform import resize
import random
from keras.preprocessing.image import ImageDataGenerator


# 二值化函数
def mybinfun(image):
    shape = image.shape
    image = np.round((255 * image)).clip(0, 255).astype(np.uint8)
    
    # 自适应阈值化
    image = cv2.adaptiveThreshold(image, 255, 
                                        cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY , 29, 5)

    # 归一化图像像素值
    image = (image / 255.0)

    return image.reshape(shape)

# 添加椒盐噪声函数
def salt_noise(image):

    rows, cols = image.squeeze().shape
    min_size=1
    max_size=2
    noisy_image = np.copy(image)
    salt_img = np.zeros((rows//2, cols//2))
    noise_mask = np.random.choice([0, 1], size=salt_img.shape[:2], p=[0.02, 0.98])
    size_mask = np.random.randint(min_size, max_size+1, size=noise_mask.shape)
    salt = np.logical_and(noise_mask == 0, size_mask > 0)
    salt_img[salt] = 1
    salt_img = resize(salt_img, (rows, cols)) 
    noisy_image[salt_img>0]=1

    return noisy_image

# 添加不同粒度的高斯噪声函数
def gauss_noise(image):

    rows, cols = image.squeeze().shape

    val = random.uniform(0.012, 0.036)

    # Full resolution
    noise_im1 = np.zeros((rows, cols))
    noise_im1 = random_noise(noise_im1, mode='gaussian', var=val**2, clip=False)

    # Half resolution
    noise_im2 = np.zeros((rows//2, cols//2))
    noise_im2 = random_noise(noise_im2, mode='gaussian', var=(val*2)**2, clip=False)  # Use val*2 (needs tuning...)
    noise_im2 = resize(noise_im2, (rows, cols))  # Upscale to original image size

    # Quarter resolution
    noise_im3 = np.zeros((rows//4, cols//4))
    noise_im3 = random_noise(noise_im3, mode='gaussian', var=(val*4)**2, clip=False)  # Use val*4 (needs tuning...)
    noise_im3 = resize(noise_im3, (rows, cols))  # What is the interpolation method?

    noisy_image = noise_im1 + noise_im2 + noise_im3  # Sum the noise in multiple resolutions (the mean of noise_im is around zero).
    noisy_image = image.reshape(noisy_image.shape) + noisy_image  # Add noise_im to the input image.
    
    noisy_image = mybinfun(noisy_image)

    noisy_image = salt_noise(noisy_image)
    
    return noisy_image.reshape(image.shape)

def Augument_MNIST(N_times = 20):

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


    # 创建图像数据生成器
    datagen = ImageDataGenerator(
        rotation_range=20,                 # 随机旋转角度范围
        width_shift_range=0.2,             # 随机水平平移范围
        height_shift_range=0.2,            # 随机垂直平移范围
        shear_range=0.2,                   # 随机剪切变换范围
        zoom_range=0.2,                    # 随机缩放范围
        fill_mode='nearest',               # 填充模式
        preprocessing_function=gauss_noise # 添加噪声
    )

    # 对训练集进行数据增强
    datagen.fit(x_train)

    # 生成增广后的数据
    augmented_data_generator = datagen.flow(x_train, y_train, batch_size=60000, shuffle=True)

    augmented_images = []
    augmented_labels = []

    # 扩增 N_times 倍
    for i in range(N_times):
        print("Data Augument Epoch({}/{}) ".format(i+1,N))
        augmented_data = next(augmented_data_generator)
        augmented_images.append(augmented_data[0])
        augmented_labels.append(augmented_data[1])

    augmented_images.append(x_train)
    augmented_labels.append(y_train)

    x_train = np.concatenate(augmented_images, axis=0)
    y_train = np.concatenate(augmented_labels, axis=0)

    # 显示前N*N张训练数据
    N = 5
    fig, axs = plt.subplots(nrows=N, ncols=N)
    ax = axs.flatten()  
    for i in range(N*N):
        ax[i].imshow(x_train[i],cmap=plt.get_cmap('gray'))
        ax[i].set_title(str(np.argmax(y_train[i]))) 
        ax[i].axis('off')  
    plt.tight_layout() 
    plt.show()

    # 保存增广后的数据
    user_input = input("是否保存数据集?(y/n): ")
    if user_input == 'y':
        np.save('augmented_images.npy', x_train)
        np.save('augmented_labels.npy', y_train)

if __name__ == "__main__":
    
    Augument_MNIST()

