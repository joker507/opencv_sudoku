#引入包
import cv2
import numpy as np


# knn算法
# 制作训练集
# 训练knn

##制作训练集##
import glob as gb

img_path = gb.glob("numbers/*")
k = 0

labels = []  # 标签
samples = []  # 样本
##提取样本数字
for path in img_path:
    img = cv2.imread(path)
    #灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #高斯平滑
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #二值化
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    #提取轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ##提取数字
    height, width = img.shape[:2]  # 图片的高和宽
    #     w = width / 5#5个数字 分5份
    list1 = []  # 储存上排数字
    list2 = []  # 储存下排数字
    for cnt in contours:  # 对每一个轮廓提取

        [x, y, w, h] = cv2.boundingRect(cnt)  # 轮廓的外接矩形

        if w > 30 and h > (height / 4):  #筛选数字轮廓
            if cv2.contourArea(cnt) > 100:
                if y < (height / 2):  # 上排数字
                    list1.append([x, y, w, h])
                else:  # 下排数字
                    list2.append([x, y, w, h])
    list1_sorted = sorted(list1, key=lambda t: t[0])  # 按照list1[0]（x坐标）排序
    list2_sorted = sorted(list2, key=lambda t: t[0])

    # 提取出每一个数字所在的矩形框，作为ROI取出。
    for i in range(5):
        [x1, y1, w1, h1] = list1_sorted[i]
        [x2, y2, w2, h2] = list2_sorted[i]
        number_roi1 = gray[y1:y1 + h1, x1:x1 + w1]  # 裁剪数字上排
        number_roi2 = gray[y2:y2 + h2, x2:x2 + w2]  # 下排
        ##数据预处理
        resized_roi1 = cv2.resize(number_roi1, (20, 40))
        thresh1 = cv2.adaptiveThreshold(resized_roi1, 255, 1, 1, 11, 2)

        resized_roi2 = cv2.resize(number_roi2, (20, 40))
        thresh2 = cv2.adaptiveThreshold(resized_roi2, 255, 1, 1, 11, 2)
        # 处理完的数字图片保存到对应的数字文件夹中
        number_path1 = 'number/%s/%d' % (str(i + 1), k) + '.jpg'
        j = i + 6
        if j == 10:
            j = 0
        number_path2 = 'number/%s/%d' % (str(j), k) + '.jpg'
        k += 1

        #归一化处理
        normalized_roi1 = thresh1 / 255
        normalized_roi2 = thresh2 / 255
        cv2.imwrite(number_path1, thresh1)
        cv2.imwrite(number_path2, thresh2)

        # 保存为训练用的数据
        #展开
        samples1 = normalized_roi1.reshape((1, 800))
        samples.append(samples1[0])  # 1,2,3,4,5
        labels.append(float(i + 1))

        samples2 = normalized_roi2.reshape((1, 800))
        samples.append(samples2[0])  # 6,7,8,9,0
        labels.append(float(j))

    # 保存为训练用的数据-转换为数组
samples = np.array(samples, np.float32)
labels = np.array(labels, np.float32)
labels = labels.reshape((labels.size, 1))  # 转化为二维数组？？？
np.save('samples.npy', samples)  #
np.save('label.npy', labels)  #

# 加载样本、标签
samples = np.load('samples.npy')
labels = np.load('label.npy')

##训练knn##
# 训练knn,经过测试成功，测试方法在下方代码
knn = cv2.ml.KNearest_create()
knn.train(samples, cv2.ml.ROW_SAMPLE, labels)

# 用80个作为训练数据、20个作为测试数据
# k = 80
# train_label = labels[:k]
# train_input = samples[:k]
# test_input = samples[k:]
# test_label = labels[k:]

#         #用opencv自带的knn训练模型
# model = cv2.ml.KNearest_create()
# model.train(train_input, cv2.ml.ROW_SAMPLE,train_label)
# 用训练好的模型测试数据中的数字
# retval, results, neigh_resp, dists = model.findNearest(test_input, 1)
# string = results.ravel()

#         #输出预测值和实际标签
# print(test_label.reshape(1,len(test_label))[0])
# print(string)
