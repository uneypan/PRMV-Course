import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pycocotools import coco

# 加载MNIST数据集
from keras.datasets import mnist

(x_train, y_train), _ = mnist.load_data()

# 选择一张MNIST图像
index = np.random.randint(0, len(x_train))
mnist_image = x_train[index]
mnist_label = y_train[index]

# 去除白底
mnist_image = mnist_image.astype(np.uint8)
_, mnist_image = cv2.threshold(mnist_image, 127, 255, cv2.THRESH_BINARY)

# 随机旋转和缩放
angle = np.random.uniform(-30, 30)
scale = np.random.uniform(0.8, 1.2)
rotation_matrix = cv2.getRotationMatrix2D((14, 14), angle, scale)
mnist_image = cv2.warpAffine(mnist_image, rotation_matrix, (28, 28))

# 加载COCO数据集
coco_data = coco.COCO('path/to/annotations.json')  # 替换为您的COCO注释文件的路径

# 随机选择一张COCO图像
image_ids = coco_data.getImgIds()
coco_image_id = np.random.choice(image_ids)
coco_image_info = coco_data.loadImgs(coco_image_id)[0]

# 读取COCO图像
coco_image_path = 'path/to/images/' + coco_image_info['file_name']  # 替换为您的COCO图像路径
coco_image = cv2.imread(coco_image_path)

# 在COCO图像上叠加MNIST图像
mnist_resized = cv2.resize(mnist_image, (56, 56))
x_pos = np.random.randint(0, coco_image.shape[1] - mnist_resized.shape[1])
y_pos = np.random.randint(0, coco_image.shape[0] - mnist_resized.shape[0])

coco_image[y_pos:y_pos + mnist_resized.shape[0], x_pos:x_pos + mnist_resized.shape[1]] = mnist_resized

# 显示结果图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(mnist_image, cv2.COLOR_GRAY2RGB))
plt.title('MNIST Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(coco_image, cv2.COLOR_BGR2RGB))
plt.title('COCO Image with MNIST')

plt.show()

# 生成COCO格式的标签
coco_annotations = []

mnist_bbox = [x_pos, y_pos, x_pos + mnist_resized.shape[1], y_pos + mnist_resized.shape[0]]
coco_annotation = {
    'id': 1,
    'image_id': coco_image_id,
    'category_id': 1,  # 替换为手写数字的类别ID
    'bbox': mnist_bbox,
    'area': mnist_resized.shape[0] * mnist_resized.shape[1],
    'iscrowd': 0
}
coco_annotations.append(coco_annotation)

# 将标签保存为COCO JSON格式
coco_dataset = {
    'images': [coco_image_info],
    'annotations': coco_annotations,
    'categories': [{'id': 1, 'name': 'handwritten_digit'}]  # 替换为手写数字的类别名称和ID
}

output_json_path = 'path/to/output.json'  # 替换为输出的COCO标签JSON文件路径
with open(output_json_path, 'w') as output_file:
    output_file.write(json.dumps(coco_dataset))

print('COCO标签文件已生成:', output_json_path)
