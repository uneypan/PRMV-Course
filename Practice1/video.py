import cv2
import os
import time
import keras
import numpy as np
                         
#规定视频编码器
fourcc = cv2.VideoWriter_fourcc(*'XVID')

#规定视频输出路径，编码器，帧率，画幅
out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (640,480))


# 加载训练好的手写数字分类模型
model = keras.models.load_model('my_mnist_model.h5')

#启用外接摄像头  
cap = cv2.VideoCapture(0)                
                                        
while(cap.isOpened()):

    ret, frame = cap.read()

    if ret==True:
        
        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 11x11自适应阈值化 
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # 膨胀操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # 腐蚀操作
        eroded = cv2.erode(dilated, kernel, iterations=3)

        # 取画面正中间的部分
        h,w = 56,56
        y,x = gray.shape
        y,x = int(y/2-h/2),int(x/2-w/2)

        input = eroded

        # 提取边界框中的手写数字图像
        digit = input[y:y + h, x:x + w]

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
        cv2.putText(frame, str(prediction[0,int(digit_class)]), (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        frame[y:y+h, x:x+w,0] = input[y:y + h, x:x + w]
        frame[y:y+h, x:x+w,1] = input[y:y + h, x:x + w]
        frame[y:y+h, x:x+w,2] = input[y:y + h, x:x + w]
        # 显示处理后的帧
        cv2.imshow('Digit Detection', frame)

        if cv2.waitKey(1) == ord('q'):        #每间隔1ms判断是否有q的退出指令从键盘输入
            break
    else:
        break

#释放以及关闭进程
cap.release()

cv2.destroyAllWindows()
    



cap.release()
out.release()
cv2.destroyAllWindows()
#释放以及关闭进程