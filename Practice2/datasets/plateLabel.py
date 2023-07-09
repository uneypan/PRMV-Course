import cv2
import imageio
import numpy as np
import os
import shutil
import argparse
plateName="京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
plate_chr="#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
def allFileList(rootfile,allFile):
    folder =os.listdir(rootfile)
    for temp in folder:
        fileName = os.path.join(rootfile,temp)
        if os.path.isfile(fileName):
            allFile.append(fileName)
        else:
            allFileList(fileName,allFile)
def is_str_right(plate_name):
    for str_ in plate_name:
        if str_ not in plate_chr:
            return False
    return True

def plateLabel(rootPath,labelFile):
    palteStr=plate_chr
    print(len(palteStr))
    plateDict ={}
    for i in range(len(list(palteStr))):
        plateDict[palteStr[i]]=i
    fp = open(labelFile,"w",encoding="utf-8")
    file =[]
    allFileList(rootPath,file)
    picNum = 0
    for jpgFile in file:
        print(jpgFile)
        jpgName = os.path.basename(jpgFile)
        name =jpgName.split("_")[0]
        if " " in name:
            continue
        labelStr=" "
        if not is_str_right(name):
            continue
        strList = list(name)
        for  i in range(len(strList)):
            labelStr+=str(plateDict[strList[i]])+" "
        # while i<7:
        #     labelStr+=str(0)+" "
        #     i+=1
        picNum+=1
        # print(jpgFile+labelStr)
        fp.write(jpgFile+labelStr+"\n")
    fp.close()