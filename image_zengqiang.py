'''
代码功能：
只是批量对图像进行除了翻转之外的其他类型的数据增强，不涉及标签变化

'''
import shutil
import os
import albumentations as A
import cv2
import torch
import numpy as np
import cv2
import math
from PIL import Image
import numpy as np

original_path = r'E:\Robotic_Grasp\datasets\20221212\yolo2cornell'
after_save = r'E:\Robotic_Grasp\datasets\20221212\enhance'

# 这块暂时不动 是新建文件夹的
# classnames = os.listdir(original_path)
# for class_file in classnames:
#     # read class
#     original_image_path = os.path.join(os.path.join(original_path, class_file),'image')
#     original_label_path = os.path.join(os.path.join(original_path, class_file), 'label')
#     after_zengqinag_image = os.path.join(os.path.join(after_save, class_file), 'iamge')
#     after_zengqinag_label = os.path.join(os.path.join(after_save, class_file), 'label')
#     if os.path.exists(after_zengqinag_image) is False:
#         os.makedirs(after_zengqinag_image)
#     if os.path.exists(after_zengqinag_label) is False:
#         os.makedirs(after_zengqinag_label)

# # 原始image和label位置
# original_image = r"C:\Users\72975\Desktop\zengqiang\image1"
# original_label = r"C:\Users\72975\Desktop\zengqiang\label1" #此处的label必须是归一化后的yolo格式，因为要用这里的去进行数据增强
#
# # 增强后需要保存的iamge和label位置
# after_zengqinag_image = r"C:\Users\72975\Desktop\zengqiang\zq1_image"
# after_zengqinag_label = r"C:\Users\72975\Desktop\zengqiang\zq1_label"
# after_save = r'C:\Users\72975\Desktop\zengqiang\after\stapler'

'''
增强模式种类，即mode可取为“
      mode = RandomBrightnessContrast  #随机亮度对比度
             RGBShift  #分别平移RGB几个像素值
             HueSaturationValue 色调饱和度值
             GaussianBlur 高斯模糊
             RandomContrast 对比度
'''

def main():

    mode = 'enhance01' #此处的mode名字是为了对增强后的图片和label进行命名

    transform = A.Compose([
        # A.RandomCrop(width=450, height=450),
        # A.HorizontalFlip(p=1),
        # A.VerticalFlip(p=1),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, always_apply=False, p=0.2),
        A.RandomContrast(limit=0.2, always_apply=False, p=0.5),
        #
        A.GaussianBlur(blur_limit=5, always_apply=False, p=0.2),  # 随机模糊处理
        A.RandomBrightnessContrast(p=0.5),#随机亮度对比度
        A.RandomGamma(gamma_limit=(130, 130), eps=None, always_apply=False, p=0.5),  # 随机灰度系数
        A.GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=0.5),
        # A.Blur(blur_limit=15, always_apply=False, p=1) # 模糊处理，尽量不用
    ])

    classnames = os.listdir(original_path)
    for class_file in classnames:
        # read class
        original_image_path = os.path.join(os.path.join(original_path, class_file), 'images')
        original_label_path = os.path.join(os.path.join(original_path, class_file), 'labels_grasp')
        original_label_path0 = os.path.join(os.path.join(original_path, class_file), 'labels')
        after_zengqinag_image = os.path.join(os.path.join(after_save, class_file), 'images')
        after_zengqinag_label = os.path.join(os.path.join(after_save, class_file), 'labels_grasp')
        after_zengqinag_label0 = os.path.join(os.path.join(after_save, class_file), 'labels')
        if os.path.exists(after_zengqinag_image) is False:
            os.makedirs(after_zengqinag_image)
        if os.path.exists(after_zengqinag_label) is False:
            os.makedirs(after_zengqinag_label)
        if os.path.exists(after_zengqinag_label0) is False:
            os.makedirs(after_zengqinag_label0)

        datanames = os.listdir(original_image_path)
        for file in datanames:
            # read image
            txt_path = os.path.join(original_image_path, file)
            image = cv2.imread(txt_path)
            H, W, _ = image.shape
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 不转换就正常的， 转换了再最后保存反而不正常

            transformed = transform(image=image)
            transformed_image = transformed['image']
            filename = os.path.splitext(file)[0]  # 去掉file的后缀xml，为了下面方便保存txt命名
            #cv2.imwrite(os.path.join(after_zengqinag_image, filename + '_0.png'), transformed_image)
            cv2.imwrite(os.path.join(after_zengqinag_image, filename[:-2] + mode + '_r.png'), transformed_image)

            # 根据图像名称找到相对应的label并复制一份，重新命名
            labelname = os.path.splitext(file)[0] #去掉后缀
            old_label_path = os.path.join(original_label_path, labelname[:-1] + 'cpos.txt')
            new_label_path = os.path.join(after_zengqinag_label, labelname[:-2] + mode+'_cpos.txt')
            shutil.copy(old_label_path, new_label_path)

            # 复制一份yolo标签到新的
            labelname0 = os.path.splitext(file)[0] #去掉后缀
            old_label_path0 = os.path.join(original_label_path0, labelname0[:-1] + 'r.txt')
            new_label_path0 = os.path.join(after_zengqinag_label0, labelname0[:-2] + mode+'_r.txt')
            shutil.copy(old_label_path0, new_label_path0)

if __name__ == "__main__":
    main()
