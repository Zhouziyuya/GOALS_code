import cv2
import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as trans
from models.task1_model import UNet
from dataset_functions.task1_dataset import OCTDataset



# 预测阶段
best_model_path = './checkpoints/task1/train2.pth'
image_size = 256
test_root = '../datasets/Validation/Image'

model = UNet()
para_state_dict = torch.load(best_model_path)
model.load_state_dict(para_state_dict)
model.eval()

img_test_transforms = trans.Compose([
    trans.ToTensor(),
    trans.Resize((image_size, image_size))
])

test_dataset = OCTDataset(image_transforms=img_test_transforms,
                        image_path=test_root,
                        mode='test')
cache = []
for img, idx, h, w in test_dataset:
    # print('./submission/task1/Layer_Segmentations-val/'+idx)
    img = img.unsqueeze(0) # 增加维度：(3,256,256)-->(1,3,256,256)
    logits = model(img) # 模型输出维度(1,4,256,256)
    m = torch.nn.Softmax(dim=1)
    logits = m(logits)
    print(logits.shape)
    pred_img = logits.detach().numpy().argmax(1)
    pred_gray = np.squeeze(pred_img, axis=0)
    pred_gray = pred_gray.astype('float32')

    pred_gray[pred_gray == 1] = 80
    pred_gray[pred_gray == 2] = 160
    pred_gray[pred_gray == 3] = 255
    pred = cv2.resize(pred_gray, (w, h))
    cv2.imwrite('./submission/task1/Layer_Segmentations-val/'+idx, pred)


# # 测试数据集
# filelists = os.listdir('./submission/task1/Layer_Segmentations-val/')
# print(filelists[0])
# img = cv2.imread(os.path.join('./submission/task1/Layer_Segmentations-val/', filelists[0]))
# print(img)
