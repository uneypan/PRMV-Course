'''
CCPD数据集压缩包解压和训练集划分脚本
CCPD数据集下载地址: https://github.com/detectRecog/CCPD
总共有2个压缩包 CCPD2019.tar.xz 和 CCPD2020.zip
在脚本目录的上一级目录新建一个CCPD文件夹, 
将 CCPD2019.tar.xz 和 CCPD2020.zip 放在里面
然后运行此脚本解压和划分训练集,
默认按照 训练:验证:测试=8:1:1 随机分配
'''

import os

def list_file_path(filesdir,endwith=None):
    
    paths=[]
    for root, dirs, files in os.walk(filesdir):
        for name in files: 
            if endwith == None:                    # 匹配文件后缀
                path = os.path.join(root, name)
                paths.append(path)
            elif name.endswith(endwith):               # 匹配文件后缀
                path = os.path.join(root, name)
                paths.append(path)
    
    return paths

import tarfile
if list_file_path('../CCPD/CCPD2019',endwith='jpg') == []:
    tar = tarfile.open('../CCPD/CCPD2019.tar.xz')
    tar.extractall('../CCPD',)
    tar.close()

import zipfile
if list_file_path('../CCPD/CCPD2020',endwith='jpg') == []:
    f = zipfile.ZipFile("../CCPD/CCPD2020.zip",'r')
    for file in f.namelist():
        f.extract(file,"../CCPD")             
    f.close()


picslist = list_file_path('../CCPD',endwith='.jpg')
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(picslist, test_size=0.1)
train_set, val_set = train_test_split(train_set, test_size=0.1/(1-0.1))
print(len(train_set),len(val_set),len(test_set))

def newPath(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path

newPath('../CCPD_YOLO/images/train/')
newPath('../CCPD_YOLO/images/val/')
newPath('../CCPD_YOLO/images/test/')
newPath('../CCPD_YOLO/labels/train/')
newPath('../CCPD_YOLO/labels/val/')
newPath('../CCPD_YOLO/labels/test/')

import os
from shutil import copy2
for p in train_set:
    copy2(p,'../CCPD_YOLO/images/train/')
    print(p)
for p in val_set:
    copy2(p,'../CCPD_YOLO/images/val/')
    print(p)
for p in test_set:
    copy2(p,'../CCPD_YOLO/images/test/')
    print(p)

