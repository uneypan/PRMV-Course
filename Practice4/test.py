
import cv2
import numpy as np
from enhance import image_enhance
from thinning import thinning
from feature import feature 




if __name__ == "__main__":
        
    img_path = "DB3_B/101_1.tif"

    if img_path:
        print(img_path)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow("img",img)


        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图

        rows, cols = np.shape(img)
        aspect_ratio = np.double(rows) / np.double(cols)

        new_rows = 200  # randomly selected number
        new_cols = new_rows / aspect_ratio

        img = cv2.resize(img, (int(new_cols), int(new_rows)))

        img = image_enhance(img)

        img = thinning(img)


        feat = feature(img)


        cv2.waitKey(0)
