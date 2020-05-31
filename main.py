import cv2
import matplotlib.pyplot as plt
import numpy as np

import get_img
import knn

# 识别图片解数独
def show(name,image):
    cv2.imshow(name,image)
    cv2.waitKeyEx()
    cv2.destroyAllWindows()

img = cv2.imread(get_img.img)
# 因为对图像进行裁剪
img = img[81:495, 10:449]

##灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

##二值化
ret, thresh = cv2.threshold(gray, 200, 255, 1)
##对图像膨胀
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
dilated = cv2.dilate(thresh, kernel)

##提取轮廓
contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 提取小方格，层级关系， 其父轮廓为0号轮廓的轮廓
boxes = []
for i in range(len(hierarchy[0])):
    if hierarchy[0][i][3] == 0:  ##判断是否为格子的轮廓，父轮廓为0，外面的大轮廓
        boxes.append(hierarchy[0][i])

    # 提取数字->装入矩阵
height, width = img.shape[:2]
box_h = height / 9#9宫格
box_w = width / 9
sudoku_arry = np.zeros((9, 9), np.int32)  # 数独数组

## 提取数字，其父轮廓都存在子轮廓
number_boxes = []
for j in range(len(boxes)):
    if boxes[j][2] != -1:  # 含有数字
        # number_boxes.append(boxes[j])
        x, y, w, h = cv2.boundingRect(contours[boxes[j][2]])
        number_boxes.append([x, y, w, h])
        img = cv2.rectangle(img, (x - 1, y - 1), (x + w + 1, y + h + 1), (0, 0, 255), 2)#框出数字
        ##对提取的数字进行处理
        number_roi = gray[y:y + h, x:x + w]
        # 统一大小
        resized_roi = cv2.resize(number_roi, (20, 40))
        ##二值化
        thresh1 = cv2.adaptiveThreshold(resized_roi, 255, 1, 1, 11, 2)
        ##归一化
        normalized_roi = thresh1 / 255
        ## 展开成一行
        sample1 = normalized_roi.reshape((1, 800))
        sample1 = np.array(sample1, np.float32)

        ## knn识别
        retval, results, neigh_resp, dists = knn.knn.findNearest(sample1, 1) #详细看knn.py
        number = int(results.ravel()[0])  # ravel()拉平为一维数组，提取数字？？

        ## 识别结果展示
        cv2.putText(img, str(number), (x - 3, y + h), 1, 3, (255, 0, 0), 2, cv2.LINE_AA)  # 画数字

        ##求数在数独矩阵中的位置，构建数独数组
        sudoku_arry[int(y / box_h), int(x / box_w)] = number

print(sudoku_arry)

#数独求解
import sudoku #代码在另外一个文件
sudoku.solve(sudoku_arry)
print('result：')
print(sudoku_arry)
re_img = img.copy()
#将解填到图片中
for y in range(9):
    for x in range(9):
        cv2.putText(re_img,str(sudoku_arry[y][x]),(int(x*box_w + 5),int(y*box_h + 40)),1, 3, (255, 0, 0), 2, cv2.LINE_AA)
solve_path = 'image/solve' + str(get_img.number) + '.jpg'
cv2.imwrite(solve_path,img) #保存图片在image文件架中看

re = np.hstack([img,re_img])
show('result',re)