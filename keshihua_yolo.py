'''
功能:可视化yolo格式的标签
data：2022/6/16
'''

import os

import albumentations as A
import cv2
import torch
import numpy as np
import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np


# 增强后保存的iamge和label位置
data_root = r"D:\yolov8_younger\Jacquard\Jacquard_Dataset_0\1a0312faac503f7dc2c1a442b53fa053"

keshihua_save = r'D:\yolov8_younger\lshujuji\keshihua_yolo'
if os.path.exists(keshihua_save) is False:
    os.makedirs(keshihua_save)

def cv_show(name,img):  #定义图片显示的方式
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():

    # # 原始image和label位置
    after_zengqinag_image = os.path.join(data_root)
    after_zengqinag_label = os.path.join(data_root)

    datanames = os.listdir(after_zengqinag_image)
    for file in datanames:
        # read image
        image_path = os.path.join(after_zengqinag_image, file)
        img = cv2.imread(image_path)

        # 根据图像名去找到该图像的标签
        labelname = os.path.splitext(file)[0]
        #labelname = labelname[0:-1]
        label_path0 = os.path.join(after_zengqinag_label, labelname + '.txt')
        with open(label_path0) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                labels = f.readline()
                if not labels:
                    break  # EOF  这句的意思就是如果labels为假，就执行break。也就是最后数据读完了，退出

                # 转换为四顶点格式（yolo是标准化后的，先回到640，480，再画框）
                labels = labels.split(';')
                labels = list(map(float, labels))
                center_x,center_y,w,h = int(labels[1]*640),int(labels[2]*480),int(labels[3]*640),int(labels[4]*480)
                p0 = center_x - w/2, center_y - h/2
                p1 = center_x + w/2, center_y - h/2
                p2 = center_x + w/2,center_y + h/2
                p3 = center_x - w/2,center_y + h/2
                a1 = (int(p0[0]),int(p0[1]))
                a2 = (int(p1[0]),int(p1[1]))
                a3 = (int(p2[0]),int(p2[1]))
                a4 = (int(p3[0]),int(p3[1]))

                # 颜色格式是按照bgr的顺序存储的
                color_black = (0, 0, 0)
                color_red = (0, 0, 255)
                color_yellow = (255, 0, 255)
                color_blue = (255, 0, 0)
                # cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
                cv2.line(img, a1, a2, color_blue, thickness=None, lineType=None, shift=None)  # 黑色
                cv2.line(img, a2, a3, color_blue, thickness=None, lineType=None, shift=None)  # 蓝色
                cv2.line(img, a3, a4, color_blue, thickness=None, lineType=None, shift=None)  # 黄色
                cv2.line(img, a4, a1, color_blue, thickness=None, lineType=None, shift=None)  # 红色

                cv2.imwrite(os.path.join(keshihua_save, file), img)
                 #cv_show('img', img)



if __name__ == "__main__":
    main()

