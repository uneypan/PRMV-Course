# -*- coding: utf-8 -*-
from math import *
import cv2
import numpy as np

def feature(img):
    
    # endpoint1 = img
    # endpoint2 = img
    features = []
    # endpoint = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if img[i, j] == 0:  # 像素点为黑
                m = i
                n = j

                eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1], img[m, n - 1], img[m, n + 1],
                              img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]

                if sum(eightField) / 255 == 7:  # 黑色块1个，端点

                    # 判断是否为指纹图像边缘
                    if sum(img[:i, j]) == 255 * i or sum(img[i + 1:, j]) == 255 * (w - i - 1) or sum(
                            img[i, :j]) == 255 * j or sum(img[i, j + 1:]) == 255 * (h - j - 1):
                        continue
                    canContinue = True
                    # print(m, n)
                    coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1], [m + 1, n - 1],
                                  [m + 1, n], [m + 1, n + 1]]
                    for o in range(8):  # 寻找相连接的下一个点
                        if eightField[o] == 0:
                            index = o
                            m = coordinate[o][0]
                            n = coordinate[o][1]
                            # print(m, n, index)
                            break
                    # print(m, n, index)
                    for k in range(4):
                        coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1],
                                      [m + 1, n - 1], [m + 1, n], [m + 1, n + 1]]
                        eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1], img[m, n - 1], img[m, n + 1],
                                      img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                        if sum(eightField) / 255 == 6:  # 连接点
                            for o in range(8):
                                if eightField[o] == 0 and o != 7 - index:
                                    index = o
                                    m = coordinate[o][0]
                                    n = coordinate[o][1]
                                    # print(m, n, index)
                                    break
                        else:
                            # print("False", i, j)
                            canContinue = False
                    if canContinue:

                        if n - j != 0:
                            if i - m >= 0 and j - n > 0:
                                direction = atan((i - m) / (n - j)) + pi
                            elif i - m < 0 and j - n > 0:
                                direction = atan((i - m) / (n - j)) - pi
                            else:
                                direction = atan((i - m) / (n - j))
                        else:
                            if i - m >= 0:
                                direction = pi / 2
                            else:
                                direction = -pi / 2
                        feature = []
                        feature.append(i)
                        feature.append(j)
                        feature.append("endpoint")
                        feature.append(direction)
                        features.append(feature)

                elif sum(eightField) / 255 == 5:  # 黑色块3个，分叉点
                    coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1], [m + 1, n - 1],
                                  [m + 1, n], [m + 1, n + 1]]
                    junctionCoordinates = []
                    junctions = []
                    canContinue = True
                    # 筛除不符合的分叉点
                    for o in range(8):  # 寻找相连接的下一个点
                        if eightField[o] == 0:
                            junctions.append(o)
                            junctionCoordinates.append(coordinate[o])
                    for k in range(3):
                        if k == 0:
                            a = junctions[0]
                            b = junctions[1]
                        elif k == 1:
                            a = junctions[1]
                            b = junctions[2]
                        else:
                            a = junctions[0]
                            b = junctions[2]
                        if (a == 0 and b == 1) or (a == 1 and b == 2) or (a == 2 and b == 4) or (a == 4 and b == 7) or (
                                a == 6 and b == 7) or (a == 5 and b == 6) or (a == 3 and b == 5) or (a == 0 and b == 3):
                            canContinue = False
                            break

                    if canContinue:  # 合格分叉点
                        # print(junctions)
                        print(junctionCoordinates)
                        print(i, j, "合格分叉点")
                        directions = []
                        canContinue = True
                        for k in range(3):  # 分三路进行
                            if canContinue:
                                junctionCoordinate = junctionCoordinates[k]
                                m = junctionCoordinate[0]
                                n = junctionCoordinate[1]
                                print(m, n, "start")
                                eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1], img[m, n - 1],
                                              img[m, n + 1],
                                              img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                                coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1],
                                              [m + 1, n - 1], [m + 1, n], [m + 1, n + 1]]
                                canContinue = False
                                for o in range(8):
                                    if eightField[o] == 0:
                                        a = coordinate[o][0]
                                        b = coordinate[o][1]
                                        print("a=", a, "b=", b)
                                        # print("i=", i, "j=", j)
                                        if (a != i or b != j) and (
                                                a != junctionCoordinates[0][0] or b != junctionCoordinates[0][1]) and (
                                                a != junctionCoordinates[1][0] or b != junctionCoordinates[1][1]) and (
                                                a != junctionCoordinates[2][0] or b != junctionCoordinates[2][1]):
                                            index = o
                                            m = a
                                            n = b
                                            canContinue = True
                                            print(m, n, index, "支路", k)
                                            break
                                if canContinue:  # 能够找到第二个支路点
                                    for p in range(3):
                                        coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1],
                                                      [m, n + 1],
                                                      [m + 1, n - 1], [m + 1, n], [m + 1, n + 1]]
                                        eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1],
                                                      img[m, n - 1],
                                                      img[m, n + 1],
                                                      img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                                        if sum(eightField) / 255 == 6:  # 连接点
                                            for o in range(8):
                                                if eightField[o] == 0 and o != 7 - index:
                                                    index = o
                                                    m = coordinate[o][0]
                                                    n = coordinate[o][1]
                                                    print(m, n, index, "支路尾")
                                                    # print(m, n, index)
                                                    break
                                        else:
                                            # print("False", i, j)
                                            canContinue = False
                                if canContinue:  # 能够找到3个连接点

                                    if n - j != 0:
                                        if i - m >= 0 and j - n > 0:
                                            direction = atan((i - m) / (n - j)) + pi
                                        elif i - m < 0 and j - n > 0:
                                            direction = atan((i - m) / (n - j)) - pi
                                        else:
                                            direction = atan((i - m) / (n - j))
                                    else:
                                        if i - m >= 0:
                                            direction = pi / 2
                                        else:
                                            direction = -pi / 2
                                    # print(direction)
                                    directions.append(direction)
                        if canContinue:
                            feature = []
                            feature.append(i)
                            feature.append(j)
                            feature.append("bifurcation")
                            feature.append(directions)
                            features.append(feature)

    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for m in range(len(features)):
        center = (int(features[m][1]), int(features[m][0]))
        if features[m][2] == "endpoint":
            
            cv2.circle(img, center, 3, (0, 255, 0), 1)
        else:
            cv2.circle(img, (int(features[m][1]), int(features[m][0])), 3, (0, 0, 255), 1)
    
    cv2.imwrite("feature.jpg",img)

    return img,features