from imutils import paths
import numpy as np
import random
import cv2
import os

from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont





CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

provincelist = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"]

wordlist = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}




class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            for el in paths.list_images(img_dir[i]):
                self.img_paths += [el]
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        filename = self.img_paths[index]
        basename = os.path.basename(filename)
        area,tilt,box,points,label,brightness,blurriness = basename.split('.')[0].split('-')
        # --- 边界框信息
        box = box.split('_')
        box = [list(map(int, i.split('&'))) for i in box]

        # --- 关键点信息
        points = points.split('_')
        points = [list(map(int, i.split('&'))) for i in points]
        # 将关键点的顺序变为从左上顺时针开始
        points = points[-2:]+points[:2]

        # --- 读取车牌号
        label = label.split('_')
        # 省份缩写
        province = provincelist[int(label[0])]
        # 车牌信息
        words = [wordlist[int(i)] for i in label[1:]]
        # 车牌号
        chars = province+''.join(words)
        label = []
        for c in chars:
            label.append(CHARS_DICT[c])

        Image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
        # Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
        Image = Image[box[0][1]:box[1][1],box[0][0]:box[1][0],:]
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        
        # im = cv2.rectangle(Image,pt1=box[0],pt2=box[1],color=(255,255,255))
        # im = cv2.putText(im,chars,(50,150),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        # cv2.imshow(" ",im)
        # cv2.waitKey(0)
        
        Image = self.PreprocFun(Image)

        if len(label) == 8:
            if self.check(label) == False:
                print(chars, basename)
                assert 0, "Error label ^~^!!!"

        # label = list() 
        # imgname, suffix = os.path.splitext(basename)
        # imgname = imgname.split("-")[0].split("_")[0]       
        # for c in imgname:
        #     label.append(CHARS_DICT[c])
   
        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        else:
            return True
