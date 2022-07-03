# task2训练代码
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch

# 设置参数
image_file = '../datasets/Train/Image'
gt_file = '../datasets/Train/Layer_Masks'
image_size = 256 # 统一输入图像尺寸
val_ratio = 0.2 # 验证/训练图像划分比例
batch_size = 8
iters = 3000
init_lr = 1e-3

# 训练集、验证集划分
filelists = os.listdir(image_file)
train_filelists, val_filelists = train_test_split(filelists, test_size=val_ratio, random_state=42)
print('Total Nums: {}, train: {}, val: {}'.format(len(filelists), len(train_filelists), len(val_filelists)))

