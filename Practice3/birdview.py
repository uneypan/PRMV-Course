# -*- encoding: UTF-8 -*-
import numpy as np
import cv2

def transform_birdview(image, P, output_shape):

    # 构建逆变换矩阵，从图像平面转换回物理平面
    P_inv = np.linalg.inv(P)

    # 进行鸟瞰图变换
    output_image = cv2.warpPerspective(image, P_inv, output_shape)

    return output_image


def estimate_pwu(U, W):
    '''
    物理平面与去畸变图像之间的射影变换矩阵P_W→U
    使用最小二乘法求解
    U和W分别是图像平面和物理平面上的对应点集
    '''
    # 构建增广矩阵A
    num_points = U.shape[0]
    A = np.zeros((2*num_points, 9))
    for i in range(num_points):
        x_U, y_U = U[i]
        x_W, y_W = W[i]
        A[2*i] = [x_W, y_W, 1, 0, 0, 0, -x_U*x_W, -x_U*y_W, -x_U]
        A[2*i+1] = [0, 0, 0, x_W, y_W, 1, -y_U*x_W, -y_U*y_W, -y_U]

    # 使用奇异值分解求解最小二乘解
    _, _, V = np.linalg.svd(A)
    P = V[-1].reshape(3, 3)

    # 标准化射影变换矩阵
    P /= P[2, 2]

    return P

def process_video(input_file, output_file, P, region,output_shape):
    # 打开视频文件
    cap = cv2.VideoCapture(input_file)

    # 获取视频帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建输出视频对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, output_shape)

    while True:
        # 读取视频帧
        ret, frame = cap.read()

        if not ret:
            break

        # 将帧转换为鸟瞰图
        birdseye_frame = transform_birdview(frame, P, region)
        birdseye_frame = cv2.resize(birdseye_frame, output_shape)

        # 写入输出视频
        out.write(birdseye_frame)

        # 显示处理进度
        # cv2.imshow('Birdseye Video', birdseye_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def undisort_image(input_image):

    fx = 1.427252100391107e+03
    fy = 1.427603685869904e+03
    cx = 6.384378928937760e+02
    cy = 3.473473948038598e+02
    k1 = 0.085197635059269
    k2 = -0.786061814065933
    k3 = 3.697077368192353
    p1 = 0
    p2 = 0

    # 转换Matlab相机矩阵为OpenCV相机矩阵
    opencv_camera_matrix = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])

    # 将畸变系数转换为OpenCV格式
    opencv_dist_coeffs = np.array([k1, k2, p1, p2, k3])

    # 去除畸变
    undistorted_image = cv2.undistort(input_image, opencv_camera_matrix, opencv_dist_coeffs)
    

    return undistorted_image


if __name__ == "__main__":

    # 定义图像平面上的点
    U = np.array([[140, 529],
                [207, 524],
                [385, 660],
                [472, 648], 
                ],dtype=np.float32)

    # 定义物理平面上的点,单位为cm
    W = np.array([[60, 25],
                [100, 25],
                [60, 350],
                [100, 350]],dtype=np.float32)
    # 计算射影变换矩阵
    P = estimate_pwu(U, W)
    print(P)

    # 处理视频 转换为鸟瞰图画幅大小为10m*4m，然后输出为1000x400的视频
    # process_video('car2.mp4', 'car2out.mp4', P, (10000,4000),(1000,500))

    # 加载输入图像
    input_image = cv2.imread('car2.png')

    fx = 1.427252100391107e+03
    fy = 1.427603685869904e+03
    cx = 6.384378928937760e+02
    cy = 3.473473948038598e+02
    k1 = 0.085197635059269
    k2 = -0.786061814065933
    k3 = 3.697077368192353
    p1 = 0
    p2 = 0

    # 转换Matlab相机矩阵为OpenCV相机矩阵
    opencv_camera_matrix = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])

    # 将畸变系数转换为OpenCV格式
    opencv_dist_coeffs = np.array([k1, k2, p1, p2, k3])

    # 去除畸变
    undistorted_image = cv2.undistort(input_image, opencv_camera_matrix, opencv_dist_coeffs)

    # 如果需要，可以进一步调整图像大小，确保没有黑边
    # 例如，使用cv2.getOptimalNewCameraMatrix()和cv2.undistort()结合来处理图像大小调整

    # 将图像转换为鸟瞰图画幅大小为10m*4m
    birdview_image = transform_birdview(undistorted_image, P,(1000,400))


    # 显示结果
    cv2.imshow("Original Image", input_image)
    cv2.imshow("Undistorted Image", undistorted_image)
    cv2.imshow("Birdview Image", birdview_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

