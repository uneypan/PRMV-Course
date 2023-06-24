# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from keras import models
import tensorflow as tf




# 反相灰度图，将黑白阈值颠倒
def accessPiexl(img):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
       for j in range(width):
           img[i][j] = 255 - img[i][j]
    return img

# 反相二值化图像
def accessBinary(img, threshold=128):
    img = accessPiexl(img)
    # 边缘膨胀，不加也可以
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
    return img


# 根据长向量找出顶点
def extractPeek(array_vals, min_vals=10, min_rect=20):
    extrackPoints = []
    startPoint = None
    endPoint = None
    for i, point in enumerate(array_vals):
        if point > min_vals and startPoint == None:
            startPoint = i
        elif point < min_vals and startPoint != None:
            endPoint = i

        if startPoint != None and endPoint != None:
            extrackPoints.append((startPoint, endPoint))
            startPoint = None
            endPoint = None

    # 剔除一些噪点
    for point in extrackPoints:
        if point[1] - point[0] < min_rect:
            extrackPoints.remove(point)
    return extrackPoints

# 寻找边缘，返回边框的左上角和右下角（利用直方图寻找边缘算法（需行对齐））
def findBorderHistogram(path):
    borders = []
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    # 行扫描
    hori_vals = np.sum(img, axis=1)
    hori_points = extractPeek(hori_vals)
    # 根据每一行来扫描列
    for hori_point in hori_points:
        extractImg = img[hori_point[0]:hori_point[1], :]
        vec_vals = np.sum(extractImg, axis=0)
        vec_points = extractPeek(vec_vals, min_rect=0)
        for vect_point in vec_points:
            border = [(vect_point[0], hori_point[0]), (vect_point[1], hori_point[1])]
            borders.append(border)
    return borders

# 寻找边缘，返回边框的左上角和右下角（利用cv2.findContours）
def findBorderContours(path, maxArea=50):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    borders = []
    for contour in contours:
        # 将边缘拟合成一个边框
        x, y, w, h = cv2.boundingRect(contour)
        if w*h > maxArea:
            border = [(x, y), (x+w, y+h)]
            borders.append(border)
    return borders


# 显示结果及边框
def showResults(path, borders, results=None):
    img = cv2.imread(path)
    # 绘制
    print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        if results:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        #cv2.circle(img, border[0], 1, (0, 255, 0), 0)
    cv2.imshow('test', img)
    cv2.waitKey(0)

# 根据边框转换为MNIST格式
def transMNIST(path, borders, size=(28, 28)):
    imgData = np.zeros((len(borders), size[0], size[0], 1), dtype='uint8')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    for i, border in enumerate(borders):
        borderImg = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
        # 根据最大边缘拓展像素
        extendPiexl = (max(borderImg.shape) - min(borderImg.shape)) // 2
        targetImg = cv2.copyMakeBorder(borderImg, 7, 7, extendPiexl + 7, extendPiexl + 7, cv2.BORDER_CONSTANT)
        targetImg = cv2.resize(targetImg, size)
        targetImg = np.expand_dims(targetImg, axis=-1)
        imgData[i] = targetImg
    return imgData

# 预测手写数字
def predict(modelpath, imgData):
    
    my_mnist_model = models.load_model(modelpath)
    print(my_mnist_model.summary())
    img = imgData.astype('float32') / 255
    results = my_mnist_model.predict(img)
    result_number = []
    for result in results:
        result_number.append(np.argmax(result))
    return result_number

if __name__ == '__main__':

    # 加载训练好的手写数字分类模型
    model = tf.keras.models.load_model('my_mnist_model.h5')

    #启用外接摄像头  
    cap = cv2.VideoCapture(0)                
                                          
    while(cap.isOpened()):

        ret, frame = cap.read()

        if ret==True:
            
            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 对图像进行阈值处理，使手写数字更加突出
            _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

            # 查找轮廓
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 遍历每个轮廓
            for contour in contours:
                # 计算轮廓的边界框
                x, y, w, h = cv2.boundingRect(contour)
                 # 根据需求定义大小范围
                width_range = [10,50]
                height_range = [10,50]

                # 检查边界框的大小是否在范围内
                if width_range[0] <= w <= width_range[1] and height_range[0] <= h <= height_range[1]:

                    # 提取边界框中的手写数字图像
                    digit = thresholded[y:y + h, x:x + w]

                    # 调整图像大小为模型所需的输入大小
                    digit = cv2.resize(digit, (28, 28))

                    # 归一化图像像素值
                    digit = digit / 255.0

                    # 扩展维度以匹配模型输入
                    digit = np.expand_dims(digit, axis=0)

                    # 使用模型进行预测
                    prediction = model.predict(digit)
                    digit_class = np.argmax(prediction)

                    # 在原始图像中绘制识别结果
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, str(digit_class), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 显示处理后的帧
            cv2.imshow('Digit Detection', frame)

            if cv2.waitKey(1) == ord('q'):        #每间隔1ms判断是否有q的退出指令从键盘输入
                break
        else:
            break

    #释放以及关闭进程
    cap.release()

    cv2.destroyAllWindows()
    


    


