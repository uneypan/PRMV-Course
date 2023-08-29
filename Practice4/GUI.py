import PySimpleGUI as sg
import cv2
import numpy as np
from enhance import ridge_segment,ridge_orient,orientation_field,ridge_freq,ridge_filter
from thinning import thinning
from feature import feature 



def normalize_image(image):
    # 归一化处理转uint8方便显示
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    return normalized_image.astype(np.uint8)

def one_step(image,window):
    # 一键处理函数
    normim, mask = ridge_segment(image, blksze = 16, thresh = 0.1)  # normalise the image and find a ROI
    window['norm'].update(data=cv2.imencode('.png', normalize_image(normim))[1].tobytes())
    
    orientim = ridge_orient(im = normim, gradientsigma = 1, blocksigma = 7, orientsmoothsigma = 7)  # find orientation of every pixel
    color_image = ((orientim / np.pi) * 255).astype(np.uint8)
    color_image = cv2.applyColorMap(color_image, cv2.COLORMAP_BONE)
    window['orim'].update(data=cv2.imencode('.png', color_image)[1].tobytes())
    
    orientafld = orientation_field(normalize_image(normim),orientim)
    window['orifld'].update(data=cv2.imencode('.png', orientafld)[1].tobytes())
    
    freq, medfreq = ridge_freq(im = normim, mask = mask, orient = orientim, blksze = 12, windsze = 5, minWaveLength = 5,maxWaveLength = 15)  # find the overall frequency of ridges
    window['freq'].update(data=cv2.imencode('.png', normalize_image(freq))[1].tobytes())
    
    newim = ridge_filter(im = normim, orient = orientim, freq = medfreq * mask, kx = 0.5, ky = 0.5)  # create gabor filter and do the actual filtering
    img = 255 * (newim >= -3)
    window['enhance'].update(data=cv2.imencode('.png', normalize_image(img))[1].tobytes())
    
    thinnim = thinning(img,num=5)
    window['thin'].update(data=cv2.imencode('.png', normalize_image(thinnim))[1].tobytes())
    
    feat = feature(thinnim)
    window['feat'].update(data=cv2.imencode('.png', normalize_image(feat))[1].tobytes())

def create_gui():
    sg.theme('DefaultNoMoreNagging')

    # 定义布局
    layout = [
        [sg.Column([[sg.Image(filename='', key='origin')],[sg.Button('原始图像')],[sg.Image(filename='', key='freq')],[sg.Button('频率图')]]),
         sg.Column([[sg.Image(filename='', key='norm')],[sg.Button('归一化')],[sg.Image(filename='', key='enhance')],[sg.Button('增强图')]]),
         sg.Column([[sg.Image(filename='', key='orim')],[sg.Button('方向图')],[sg.Image(filename='', key='thin')],[sg.Button('细化图')]]),
         sg.Column([[sg.Image(filename='', key='orifld')],[sg.Button('方向场')],[sg.Image(filename='', key='feat')],[sg.Button('特征图')]])],
        [sg.Button('选择图片'),sg.Button('一键处理'),sg.Button('退出')],
    ]
    
    # 创建窗口
    window = sg.Window('指纹特征提取交互界面', layout, finalize=True,)
    
    # 200x200空白填充
    d = cv2.imencode('1.png', np.ones((200, 200), dtype=np.uint8)*255)[1].tobytes()
    for key in ['origin','norm','orim','orifld','freq','enhance','thin','feat']:
        window[key].update(data=d)

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, '退出'):
            break
        elif event == '选择图片':
            file_path = sg.popup_get_file('选择图片文件', file_types=(("Image Files", "*.png *.jpg *.jpeg *.bmp"),))
            if file_path:
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (200, 200))
                window['origin'].update(data=cv2.imencode('.png', image)[1].tobytes())
                processed_images = [image] * 8  # 初始时所有处理结果都是原图
        elif event == '一键处理':
            if 'image' not in locals():
                sg.popup('请先选择图片')
                continue
            one_step(image,window)
        elif event in ('原始图像','归一化', '方向图', '方向场', '频率图', '增强图', '细化图', '特征图'):
            if 'image' in locals():  # 确保已选择图片
                index = ['原始图像','归一化', '方向图', '方向场', '频率图', '增强图', '细化图', '特征图'].index(event)

                if event == '归一化':
                    normim, mask = ridge_segment(image, blksze = 16, thresh = 0.1)  # normalise the image and find a ROI
                    processed_images[index] = normalize_image(normim)
                    window['norm'].update(data=cv2.imencode('.png', processed_images[index])[1].tobytes())
                elif event == '方向图':
                    if 'normim' not in locals():
                        sg.popup('请先进行归一化处理')
                        continue
                    orientim = ridge_orient(im = normim, gradientsigma = 1, blocksigma = 7, orientsmoothsigma = 7)  # find orientation of every pixel
                    color_map = cv2.COLORMAP_BONE # 色彩映射
                    color_image = ((orientim / np.pi) * 255).astype(np.uint8)
                    color_image = cv2.applyColorMap(color_image, color_map)
                    processed_images[index] = color_image
                    window['orim'].update(data=cv2.imencode('.png', processed_images[index])[1].tobytes())
                elif event == '方向场':
                    if 'orientim' not in locals():
                        sg.popup('请先进行方向图像处理')
                        continue
                    orientafld = orientation_field(normalize_image(normim),orientim)
                    processed_images[index] = orientafld
                    window['orifld'].update(data=cv2.imencode('.png', processed_images[index])[1].tobytes())
                elif event == '频率图':
                    if 'orientim' not in locals():
                        sg.popup('请先进行方向图像处理')
                        continue
                    freq, medfreq = ridge_freq(im = normim, mask = mask, orient = orientim, blksze = 12, windsze = 5, minWaveLength = 5,
                               maxWaveLength = 15)  # find the overall frequency of ridges
                    processed_images[index] = normalize_image(freq)
                    window['freq'].update(data=cv2.imencode('.png', processed_images[index])[1].tobytes())
                elif event == '增强图':
                    if 'freq' not in locals():
                        sg.popup('请先进行频率图像处理')
                        continue
                    newim = ridge_filter(im = normim, orient = orientim, freq = medfreq * mask, kx = 0.65, ky = 0.65)  # create gabor filter and do the actual filtering
                    img = 255 * (newim >= -3)
                    processed_images[index] = normalize_image(img)
                    window['enhance'].update(data=cv2.imencode('.png', processed_images[index])[1].tobytes())
                elif event == '细化图':
                    if 'newim' not in locals():
                        sg.popup('请先进行增强图像处理')
                        continue
                    else:
                        thinnim = thinning(img)
                        processed_images[index] = normalize_image(thinnim)
                        window['thin'].update(data=cv2.imencode('.png', processed_images[index])[1].tobytes())
                    
                elif event == '特征图':
                    if 'thinnim' not in locals():
                        sg.popup('请先进行细化图像处理')
                        continue
                    else:
                        feat = feature(thinnim)
                        processed_images[index] = normalize_image(feat)
                        window['feat'].update(data=cv2.imencode('.png', processed_images[index])[1].tobytes())

                
                

        
    window.close()

if __name__ == '__main__':
    create_gui()
