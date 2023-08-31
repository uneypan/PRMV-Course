# -*- coding: utf-8 -*-
import PySimpleGUI as sg
import cv2
import numpy as np
from enhance import *
from thinning import thinning
from feature import feature 



def create_gui():
    sg.theme('DefaultNoMoreNagging')

    # 定义布局
    layout = [
        [
         sg.Column([[sg.Image(filename='', key='origin')],[sg.Button('原始图像')],[sg.Image(filename='', key='filted')],[sg.Button('滤波图')]]),
         sg.Column([[sg.Image(filename='', key='norm')],[sg.Button('归一化')],[sg.Image(filename='', key='enhance')],[sg.Button('增强图')]]),
         sg.Column([[sg.Image(filename='', key='orifld')],[sg.Button('方向场')],[sg.Image(filename='', key='thin')],[sg.Button('细化图')]]),
         sg.Column([[sg.Image(filename='', key='freq')],[sg.Button('频率图')],[sg.Image(filename='', key='feat')],[sg.Button('特征图')]]),
         sg.Column([[sg.Text("端点\n\n 坐标     方向（度）")],[sg.Text(" ", key="minutiae", size=(30, 15))],[sg.Text("分支点\n\n 坐标     三个分支方向（度）")],[sg.Text(" ", key="bifurcations", size=(30, 15))]])
        ],
        [sg.Button('选择图片'),sg.Button('一键处理'),sg.Button('退出')],
    ]
    
    # 创建窗口
    window = sg.Window('指纹特征提取交互界面', layout, finalize=True,)
    
    # 200x200空白填充
    d = cv2.imencode('1.png', np.ones((200, 200), dtype=np.uint8)*255)[1].tobytes()
    for key in ['origin','norm','orifld','freq','filted','enhance','thin','feat']:
        window[key].update(data=d)

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, '退出'):
            break
        elif event == '选择图片':
            file_path = sg.popup_get_file('选择图片文件', file_types=(("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif"),))
            if file_path:
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (200, 200))
                window['origin'].update(data=cv2.imencode('.png', image)[1].tobytes())

        elif event == '一键处理':
            if 'image' not in locals():
                sg.popup('请先选择图片')
                continue
            one_step(image,window)
        elif event in ('原始图像','归一化', '滤波图', '方向场', '频率图', '增强图', '细化图', '特征图'):
            if 'image' in locals():  # 确保已选择图片
                index = ['原始图像','归一化', '滤波图', '方向场', '频率图', '增强图', '细化图', '特征图'].index(event)

                if event == '归一化':
                    normim, mask = ridge_segment(image, blksze = 16, thresh = 0.1)  # normalise the image and find a ROI
                    window['norm'].update(data=cv2.imencode('.png', normalize_image(normim))[1].tobytes())
                elif event == '方向场':
                    if 'normim' not in locals():
                        sg.popup('请先进行归一化处理')
                        continue
                    orientim = ridge_orient(im = normim, gradientsigma = 1, blocksigma = 7, orientsmoothsigma = 7)  # find orientation of every pixel
                    color_map = cv2.COLORMAP_BONE # 色彩映射
                    color_image = ((orientim / np.pi) * 255).astype(np.uint8)
                    color_image = cv2.applyColorMap(color_image, color_map)
                    orientafld = orientation_field(normalize_image(normim),orientim)
                    window['orifld'].update(data=cv2.imencode('.png', orientafld)[1].tobytes())
                elif event == '频率图':
                    if 'orientim' not in locals():
                        sg.popup('请先计算方向场')
                        continue
                    freq, medfreq = ridge_freq(im = normim, mask = mask, orient = orientim, blksze = 12, windsze = 5, minWaveLength = 5,
                               maxWaveLength = 15)  # find the overall frequency of ridges
                    window['freq'].update(data=cv2.imencode('.png', normalize_image(freq))[1].tobytes())
                elif event == '滤波图':
                    if 'freq' not in locals():
                        sg.popup('请先计算频率图')
                        continue
                    newim = ridge_filter(im = normim, orient = orientim, freq = medfreq * mask, kx = 0.65, ky = 0.65)  # create gabor filter and do the actual filtering
                    window['filted'].update(data=cv2.imencode('.png', normalize_image(newim))[1].tobytes())
                elif event == '增强图':
                    if 'newim' not in locals():
                        sg.popup('请先计算滤波图')
                        continue
                    encim = enhance_Thres(newim)    
                    window['enhance'].update(data=cv2.imencode('.png', (encim.astype(np.uint8)))[1].tobytes())
                elif event == '细化图':
                    if 'newim' not in locals():
                        sg.popup('请先进行增强图像处理')
                        continue
                    else:
                        thinnim = thinning(encim,num=10)
                        window['thin'].update(data=cv2.imencode('.png', normalize_image(thinnim))[1].tobytes())
                    
                elif event == '特征图':
                    if 'thinnim' not in locals():
                        sg.popup('请先进行细化图像处理')
                        continue
                    else:
                        feat = feature(thinnim)
                        window['feat'].update(data=cv2.imencode('.png', normalize_image(feat))[1].tobytes())
        
    window.close()




def one_step(image,window):

    # 一键处理函数

    normim, mask = ridge_segment(image, blksze = 16, thresh = 0.1)  # normalise the image and find a ROI
    cv2.imwrite('normim.png',normalize_image(normim))
    window['norm'].update(data=cv2.imencode('.png', normalize_image(normim))[1].tobytes())
    

    orientim = ridge_orient(im = normim, gradientsigma = 1, blocksigma = 7, orientsmoothsigma = 7)  # find orientation of every pixel
    color_image = ((orientim / np.pi) * 255).astype(np.uint8)
    color_image = cv2.applyColorMap(color_image, cv2.COLORMAP_BONE)
    cv2.imwrite('orientim.png',color_image)
    
    orientafld = orientation_field(normalize_image(normim),orientim)
    cv2.imwrite('orientafld.png',orientafld)
    window['orifld'].update(data=cv2.imencode('.png', orientafld)[1].tobytes())
    
    freq, medfreq = ridge_freq(im = normim, mask = mask, orient = orientim, blksze = 12, windsze = 5, minWaveLength = 5, maxWaveLength = 15)  # find the overall frequency of ridges
    cv2.imwrite('freq.png',normalize_image(freq))
    window['freq'].update(data=cv2.imencode('.png', normalize_image(freq))[1].tobytes())
    
    newim = ridge_filter(im = normim, orient = orientim, freq = medfreq * mask, kx = 0.65, ky = 0.65)  # create gabor filter and do the actual filtering
    cv2.imwrite('filted.png',normalize_image(newim))
    window['filted'].update(data=cv2.imencode('.png', normalize_image(newim))[1].tobytes())

    encim = enhance_Thres(newim)
    cv2.imwrite('enhance.png',encim)    
    window['enhance'].update(data=cv2.imencode('.png', (encim.astype(np.uint8)))[1].tobytes())
    
    thinnim = thinning(encim,num=10)
    # thinnim = 255-cv2.ximgproc.thinning(255-encim)
    cv2.imwrite('thinnim.png',normalize_image(thinnim))
    window['thin'].update(data=cv2.imencode('.png', normalize_image(thinnim))[1].tobytes())
    
    featim,feats = feature(thinnim)
    cv2.imwrite('feat.png',featim)
    window['feat'].update(data=cv2.imencode('.png', featim)[1].tobytes())

    txtminutiae = ''
    txtbifurcations = ''
    for feat in feats:
        x,y,lab,direction = feat
        if lab == "endpoint":
            txtminutiae += "[{:<3},{:<3}]    {:>5.1f}\n".format(x,y,direction/np.pi*180+180)
        else:
            (d1,d2,d3) = direction
            txtbifurcations += "[{:<3},{:<3}]  {:>5.1f}，{:>5.1f}，{:>5.1f}\n".format(x,y,d1/np.pi*180+180,d2/np.pi*180+180,d3/np.pi*180+180)
        print('x:',x,'y:',y,'lab:',lab,'direction:',direction)

    window["minutiae"].update(txtminutiae)
    window["bifurcations"].update(txtbifurcations)




if __name__ == '__main__':
    create_gui()
