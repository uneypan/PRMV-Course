## 车速和撞线时间估计

一种基于图像处理和卡尔曼滤波算法的汽车测速和撞线时间估计方法。首先通过使用张正友相机标定模板对相机进行内参标定，然后通过实地测量得到停止线的四个角点坐标，使用奇异值分解求解射影变换矩阵对图像进行去畸变和逆透视变换，从而得到车辆在实际物理平面的鸟瞰图。然后，通过竖直方向帧间像素变化量统计直方图检测车辆特征位置，并利用卡尔曼滤波算法对车辆的位置和速度进行优化估计。

### 配置要求
本项目已经在 Python 3.8.16 环境下测试通过。
```
matplotlib
numpy
opencv_python == 4.7.0.72
```
可以使用以下命令安装
```
pip install -r requirements.txt
```
### 测试

**在本文件夹下，运行以下命令**来处理测试视频 *car.mp4*
```
python main.py
```
将出现原始视频和处理流程中间的图像，包括鸟瞰图、差分图、变化像素累计直方图等。

同时命令行会显示信息，类似于：
```
Nowtime: 0.44s  No feature detected.
Nowtime: 0.45s  No feature detected.
Nowtime: 0.45s  No feature detected.
...
Nowtime: 2.63s  Realtime velocity: 19.9km/h. 
...
Nowtime: 2.67s  Realtime velocity: 19.2km/h. Car hit line. 
...
Nowtime: 2.75s  Car passed. Time hit line: 2.67s and 2.74s. Average velocity: 19.2km/h. 
```

处理完成后，会显示实时车速和位置曲线。

同时保存实际速度下的视频 *output.mp4*、特征点移动视频 *plot_output.mp4* 。
