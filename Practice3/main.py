# -*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from birdview import transform_birdview, estimate_pwu, undisort_image
from KalmanFilter import KalmanFilter



# 绘制鸟瞰图范围，单位cm，按照 1 pixel/cm 的分辨率输出
region = (1200,400)
output_shape = region

# 将停止线前沿置于x=1000处
stopline_x_positon = 1000

# 将停止线宽度设置为40cm
stopline_width = 40

# 定义图像平面上的点U(像素)和物理平面上的点W（cm）
U = np.array([[1085, 474],[1180, 492],[629, 624],[717, 665], ],dtype=np.float32)
W = np.array([[0, 50],[40, 50],[0, 250],[40, 250]],dtype=np.float32)
W[:,0] = W[:,0] + stopline_x_positon

# 计算射影变换矩阵
P = estimate_pwu(U, W)
print(P)

# 加载视频
video_path = 'car.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频信息
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
fps = 240 # 由于编码问题不能自动获取，需要手动设置

#规定视频输出路径，编码器，帧率，画幅
output_path = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_path, fourcc, fps, (width,height))

#规定视频画图路径，编码器，帧率，画幅
plot_path = 'plot_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
plot_video = cv2.VideoWriter(plot_path, fourcc, fps, output_shape) 


# 定义累积直方图
THRESHOLD = 30 # 变化像素数量的阈值
histogram = np.zeros(region[0])
index = 0

# 绘制直方图
plt.figure(figsize=(region[0]/100, region[1]/100),dpi=100) # 设置画布大小
plt.plot(histogram) # 绘制直方图
# plt.plot([0,region[0]],[THRESHOLD,THRESHOLD],color='r') # 绘制阈值线
plt.plot([stopline_x_positon,stopline_x_positon],[0,region[1]],color='r') # 绘制停止线前沿
plt.plot([stopline_x_positon+stopline_width,
          stopline_x_positon+stopline_width],[0,region[1]],color='r') # 绘制停止线后沿
plt.scatter(0,THRESHOLD,marker='o',color='g') # 绘制特征点
plt.xlim([0,region[0]])
plt.ylim([0,region[1]])

# 车辆已经经过，停止测速标识位
finish_flag = False
t1,t2 = 0,0
log = {
    "time": [],
    "velocity": [],
    "x_position": [],
    "x0": [],
    "str": []
}
# 处理视频帧
while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        break
    
    # 将帧转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 去除镜头畸变
    gray = undisort_image(gray)
    # 将帧转换为鸟瞰图
    birdseye_frame = transform_birdview(gray, P, region)
    
    # # 计算当前帧与上一帧的差异
    if 'previous_frame' in locals():

        # 相邻帧差分
        diff = cv2.absdiff(birdseye_frame, previous_frame)

        # 将差异图像转换为二值图像
        _, binary_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
        # 沿竖直方向对二值图像进行求和
        # 得到变化像素数量在横向的分布
        histogram = np.sum(binary_diff, axis=0)/255
        
        # 从右往左查找第一个大于阈值的像素
        # 该像素对应的位置认为车头的位置
        x_position = 0
        velocity = 0
        for i in range(stopline_x_positon+stopline_width-1,0,-1):
            if histogram[i] > THRESHOLD:
                x_position = i
                break
        
        str = "Nowtime: {:.2f}s  ".format(index/fps)
        
        
        if x_position > 0 and finish_flag == False:      
            if 'kf' not in locals():
                # 为新特征点创建Kalman滤波器
                kf = KalmanFilter(Q1=0.001,Q2=0.001,R1=81,P1=16,P2=0.1)
                kf.x = np.asanyarray([x_position,2.5])
            else:
                # 更新滤波器
                kf.run(z=x_position)
                # 计算实时速度
                filted_position = kf.x[0]
                velocity = kf.x[1] / 100 * fps * 3.6
                str += "Realtime velocity: {:.1f}km/h. ".format(velocity)
                # 如果车头经过停止线前沿，则记录时间
                if kf.x[0] - stopline_x_positon > 0 and t1==0:
                    t1 = index/fps
                    str+="Car hit line. "
                # 如果车头位置超过停止线后沿，则停止测速
                if kf.x[0] - stopline_x_positon - stopline_width > 0 and t2==0:
                    t2 = index/fps
                    finish_flag = True
                    del kf
        elif finish_flag == True:
            str += "Car passed. "
            # 计算过线时间
            str += "Time hit line: {:.2f}s and {:.2f}s. ".format(t1,t2)
            # 计算过线平均速度
            str += "Average velocity: {:.1f}km/h. ".format(stopline_width/100/(t2-t1)*3.6)
        else:
            str +=  "No feature detected."
    
        print(str)

        log["time"].append(index/fps)
        log["velocity"].append(velocity)
        log["x_position"].append(x_position/100)
        log["str"].append(str)
        if 'kf' not in locals():
            log["x0"].append(0)
        else:
            log["x0"].append(kf.x[0]/100)

        cv2.putText(frame, str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        # plt绘图
        line = plt.gca().get_lines()[0]  # 获取绘图中的线对象
        line.set_ydata(histogram)  # 更新线对象的数据
        point = plt.gca().collections[0]  # 获取绘图中的点对象
        point.set_offsets((x_position, THRESHOLD))  # 更新点对象的数据

        # 获取绘图的像素数组
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()
        w, h = canvas.get_width_height()
        image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
        plot_image = cv2.resize(image_array, output_shape)
        plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)
        
        # # 写入新视频
        output_video.write(frame)
        plot_video.write(plot_image)
        # # 显示视频
        cv2.imshow('frame', frame)
        cv2.imshow('undisort birdview', birdseye_frame)
        cv2.imshow('changed pixels between images', binary_diff)
        cv2.imshow('histogram of changed pixels', plot_image)
        if(index==400):
            cv2.imwrite("frame.jpg",frame)
            cv2.imwrite("birdseye.jpg",birdseye_frame)
            cv2.imwrite("undisort.jpg",gray)
            cv2.imwrite("changed pixels between images.jpg",binary_diff)
            cv2.imwrite("histogram.jpg",plot_image)

        # 按q键退出 按s键暂停
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.waitKey(0)

    previous_frame = birdseye_frame.copy()

    index = index + 1

plt.close()
cap.release()
output_video.release()
plot_video.release()

plt.figure()
plt.subplot(211)
plt.plot(log["time"],log["velocity"])
plt.ylabel("Realtime Velocity (km/h)")
plt.legend(["Realtime velocity"])
plt.xlabel("Time (s)")
plt.xlim([1.33,2.73])

plt.subplot(212)
plt.plot(log["time"],log["x_position"])
plt.plot(log["time"],log["x0"])
plt.legend(["Realtime position","Kalman filtered position"])
plt.ylabel("Position (m)")
plt.xlabel("Time (s)")
plt.xlim([1.33,2.73])
plt.show()

