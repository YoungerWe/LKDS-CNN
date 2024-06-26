'''
功能：主要是对水平翻转和垂直翻转两种类型进行增强，并最后输出保存增强后的旋转框坐标信息
注：没有用到旋转函数。 就是最后用个cv2.boxPoints函数
data：2022/6/16
'''

import os
import albumentations as A
import cv2
import torch
import numpy as np
import cv2
import math

# 原始image和label位置
data_root = r'E:\Robotic_Grasp\datasets\20221212\xml2txt'  # 该目录下就是所有种类（bolt driver，hammer....）

# 翻转后需要保存的iamge和label位置
fanzhuan_label_image = r"E:\Robotic_Grasp\datasets\20221212\fanzhuan_yolo"


# original_image = r"C:\Users\72975\Desktop\dataset\dataset0\original\Hammer\image"
# original_label = r"C:\Users\72975\Desktop\dataset\dataset0\original\Hammer\label" #此处的label必须是归一化后的yolo格式，因为要用这里的去进行数据增强
#
# # 增强后需要保存的iamge和label位置
# after_zengqinag_image = r'C:\Users\72975\Desktop\dataset\dataset0\fanzhuan'
# after_zengqinag_image = r"C:\Users\72975\Desktop\dataset\dataset0\original\Hammer\bianhuan\image"
# after_zengqinag_label = r"C:\Users\72975\Desktop\dataset\dataset0\original\Hammer\bianhuan\label"


def main():
    mode = 'Hor'
    transform = A.Compose([
        # A.RandomCrop(width=450, height=450),
        A.HorizontalFlip(p=1),
        # A.VerticalFlip(p=1),
        # A.RandomBrightnessContrast(p=1),#随机亮度对比度
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    classnames = os.listdir(data_root)
    for class_file in classnames:
        # # 原始image和label位置
        original_image = os.path.join(os.path.join(data_root, class_file), 'images')
        original_label = os.path.join(os.path.join(data_root, class_file), 'labels')

        data_label = os.listdir(original_label)
        for file in data_label:
            # 图像地址  靠标签去索引照片名字 #前提是照片和label的名字除了后缀不一样，其余都一样
            imagename = os.path.splitext(file)[0]
            #imagename = imagename[0:-4]   #自己数据集用，做了相应的修改,去标label名字的cpos
            image_path = os.path.join(original_image, imagename + '.png')
            image = cv2.imread(image_path)
            # 标签地址
            label_path = os.path.join(original_label, file)
            #gt_boxes = []

            with open(label_path) as f:
                gt_boxes = []
                angle_temp = []
                class_labels = []
                while True:
                    # Load 4 lines at a time, corners of bounding box.
                    p0 = f.readline()  # 读取的一行数据，也就是抓取矩形标签的一行2个值，即一个顶点坐标
                    if not p0:
                        break  # EOF  这句的意思就是如果p0为假，就执行break。也就是最后数据读完了，退出
                    # p1, p2, p3 = f.readline(), f.readline(), f.readline()
                    # # 去掉换行符
                    p0 = p0.strip('\n')
                    p0 = p0.split()  # 转换为列表list()不行，用这个
                    p0 = list(map(float, p0))   # 一开始的p0内每个元素都是字符串，这里转成数字
                    class_labels.append(p0[0])
                    p0_temp = p0[1:]  #切片，去掉最后一个角度元素
                    #angle_temp.append(p0[4])  # 角度在数据增强的时候用不到，但是在转4顶点坐标的时候需要用到
                    gt_boxes.append(p0_temp)

            fanzhuan_image = os.path.join(os.path.join(fanzhuan_label_image, class_file), 'images')
            fanzhuan_label = os.path.join(os.path.join(fanzhuan_label_image, class_file), 'labels')

            if os.path.exists(fanzhuan_image) is False:
                os.makedirs(fanzhuan_image)
            if os.path.exists(fanzhuan_label) is False:
                os.makedirs(fanzhuan_label)

            transformed = transform(image=image, bboxes=gt_boxes, class_labels=class_labels)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']
            cv2.imwrite(os.path.join(fanzhuan_image, imagename[0:-2] + mode + '_r.png'), transformed_image)

            for j in range(len(transformed_bboxes)):  # 遍历一个抓取矩形  j是遍历一张图像总的抓取矩形总数目
                transformed_bboxes[j] = list(transformed_bboxes[j])  # 先把每个抓取框变成一个列表，不变之前是tuple
                for i in range(len(transformed_bboxes[j])):            # i是遍历一个抓取矩形内所有的元素，line就是每个抓取矩形框的
                    transformed_bboxes[j][i] = round(transformed_bboxes[j][i], 6)   # 每一个元素保留两位小数
                transformed_bboxes[j].insert(0,int(transformed_class_labels[j]))  # 这个就是把每个抓取框又加上了旋转角度

            #filename = os.path.splitext(file)[0]  #
            # 下面这个with就相当于与创建了一个新文件
            newfile = file[:-6] + mode +'_r.txt'
            with open(os.path.join(fanzhuan_label, newfile), "w") as f:
            #with open(os.path.join(after_zengqinag_label, file), "w") as f:
                index = 0
                for line0 in  transformed_bboxes:
                    info = [str(i) for i in line0]
                    if index == 0:
                        f.write(" ".join(info))
                    else:
                        f.write("\n" + " ".join(info))
                    index += 1

if __name__ == "__main__":
    main()

