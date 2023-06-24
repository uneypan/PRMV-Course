import cv2
import os
import time


cap = cv2.VideoCapture(0)                
#启用外接摄像头                                        

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#规定视频编码器

out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (640,480))
#规定视频输出路径，编码器，帧率，画幅

sum_time = 0
#初始化计时器

pic_num = 0
#初始化保存图片的后缀数

while(cap.isOpened()):



    ret, frame = cap.read()

    if ret==True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        out.write(frame)                      #写入视频流
        cv2.imshow('frame',frame)             #展示监视器

        if cv2.waitKey(1) == ord('q'):        #每间隔1ms判断是否有q的退出指令从键盘输入
            break
    else:
        break




cap.release()
out.release()
cv2.destroyAllWindows()
#释放以及关闭进程