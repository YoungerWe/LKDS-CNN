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
fanzhuan_label_image = r"E:\Robotic_Grasp\datasets\20221212\fanzhuan"


# original_image = r"C:\Users\72975\Desktop\dataset\dataset0\original\Hammer\image"
# original_label = r"C:\Users\72975\Desktop\dataset\dataset0\original\Hammer\label" #此处的label必须是归一化后的yolo格式，因为要用这里的去进行数据增强
#
# # 增强后需要保存的iamge和label位置
# after_zengqinag_image = r'C:\Users\72975\Desktop\dataset\dataset0\fanzhuan'
# after_zengqinag_image = r"C:\Users\72975\Desktop\dataset\dataset0\original\Hammer\bianhuan\image"
# after_zengqinag_label = r"C:\Users\72975\Desktop\dataset\dataset0\original\Hammer\bianhuan\label"


def main():
    mode = 'Ver'
    transform = A.Compose([
        # A.RandomCrop(width=450, height=450),
        # A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        # A.RandomBrightnessContrast(p=1),#随机亮度对比度
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # image = cv2.imread(r"C:\Users\72975\Desktop\pcd0100r.png")
    # H, W, _ = image.shape
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    classnames = os.listdir(data_root)
    for class_file in classnames:
        # # 原始image和label位置
        original_image = os.path.join(os.path.join(data_root, class_file), 'images')
        original_label = os.path.join(os.path.join(data_root, class_file), 'labels_grasp')

        data_label = os.listdir(original_label)
        for file in data_label:
            # 图像地址  靠标签去索引照片名字 #前提是照片和label的名字除了后缀不一样，其余都一样
            imagename = os.path.splitext(file)[0]
            imagename = imagename[0:-4]   #自己数据集用，做了相应的修改,去标label名字的cpos
            image_path = os.path.join(original_image, imagename + 'r.png')
            image = cv2.imread(image_path)
            # 标签地址
            label_path = os.path.join(original_label, file)
            #gt_boxes = []
            with open(label_path) as f:
                gt_boxes = []
                angle_temp = []
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
                    p0_temp = p0[0:-1]  #切片，去掉最后一个角度元素
                    angle_temp.append(p0[4])  # 角度在数据增强的时候用不到，但是在转4顶点坐标的时候需要用到
                    gt_boxes.append(p0_temp)
            class_labels = []
            for i in range(len(gt_boxes)):
                class_labels.append('yaokongqi')

            fanzhuan_image = os.path.join(os.path.join(fanzhuan_label_image, class_file), 'images')
            fanzhuan_label = os.path.join(os.path.join(fanzhuan_label_image, class_file), 'labels_grasp')

            if os.path.exists(fanzhuan_image) is False:
                os.makedirs(fanzhuan_image)
            if os.path.exists(fanzhuan_label) is False:
                os.makedirs(fanzhuan_label)

            transformed = transform(image=image, bboxes=gt_boxes, class_labels=class_labels)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']
            cv2.imwrite(os.path.join(fanzhuan_image, imagename[0:-1] + mode + '_r.png'), transformed_image)

            for j in range(len(transformed_bboxes)):  # 遍历一个抓取矩形  j是遍历一张图像总的抓取矩形总数目
                transformed_bboxes[j] = list(transformed_bboxes[j])  # 先把每个抓取框变成一个列表，不变之前是tuple
                for i in range(len(transformed_bboxes[j])):            # i是遍历一个抓取矩形内所有的元素，line就是每个抓取矩形框的
                    transformed_bboxes[j][i] = round(transformed_bboxes[j][i], 2)
                transformed_bboxes[j].append(angle_temp[j])  # 这个就是把每个抓取框又加上了旋转角度


            # 开始转换为4顶点格式
            four_point_bboxes = []
            for p in range(len(transformed_bboxes)):
                center = []

                center.append(transformed_bboxes[p][0])
                center.append(transformed_bboxes[p][1])
                center[0] = round(center[0] * 640.0, 4)
                center[1] = round(center[1] * 480.0, 4)

                w = round(transformed_bboxes[p][2] * 640.0, 4)
                h = round(transformed_bboxes[p][3] * 480.0, 4)

                # 水平翻转，就是math.pi-原来的弧度，即angle = angle = math.pi - (transformed_bboxes[p][4])
                # 垂直翻转，和水平翻转一样
                angle = math.pi - (transformed_bboxes[p][4])    # 这里是弧度
                angle = math.degrees(angle)    #弧度转角度，因为下面的cv2.boxPoints内的angle参数需要用的角度

                # 此处用到的矩形中心点center[0], center[1]就是增强库后得到的，因为增强库默认是对水平矩形框进行增强，虽然是水平框
                # 但是得到的中心点和宽高不会变，因此再做相应的旋转就可以得到，增强后的矩形坐标信息
                ro_rect = ((center[0], center[1]), (w, h), angle)  # 这里的angle是角度，不是弧度
                box0 = cv2.boxPoints(ro_rect) #一个box0得到的就是一个矩形框的4顶点坐标
                detx1 = (box0[0][0] - box0[1][0]) * (box0[0][0] - box0[1][0])
                dety1 = (box0[0][1] - box0[1][1]) * (box0[0][1] - box0[1][1])
                L1 = math.sqrt(detx1 + dety1)
                detx2 = (box0[1][0] - box0[2][0]) * (box0[1][0] - box0[2][0])
                dety2 = (box0[1][1] - box0[2][1]) * (box0[1][1] - box0[2][1])
                L2 = math.sqrt(detx2 + dety2)
                if L1 < L2:
                    temp = []
                    temp.append(box0[2][0])
                    temp.append(box0[2][1])
                    temp.append(box0[0][0])
                    temp.append(box0[0][1])

                    box0[0][0] = temp[0]
                    box0[0][1] = temp[1]
                    box0[2][0] = temp[2]
                    box0[2][1] = temp[3]
                four_point_bboxes.append(box0)

            #上面得到的four_point_bboxes是个6个元素的列表，每个元素都是个（4,2）形式的矩阵
            # 将得到的4点格式label按照1个顶点(2个坐标值)一行写入txt文件内
            #filename = os.path.splitext(file)[0]  #
            # 下面这个with就相当于与创建了一个新文件
            newfile = file[:-9] + mode +'_cpos.txt'
            with open(os.path.join(fanzhuan_label, newfile), "w") as f:
            #with open(os.path.join(after_zengqinag_label, file), "w") as f:
                for data1 in four_point_bboxes:
                    for line0 in data1:
                        for i in range(len(line0)):
                            a = round(line0[i], 2)
                            if i % 2 != 1:
                                f.write(str(a) + ' ')
                            else:
                                f.write(str(a) + '\n')

if __name__ == "__main__":
    main()

