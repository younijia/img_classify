import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from model import Wide_resnet50_2_ML_Dynamic

class_names = ['Vehicle', 'Sky', 'Food', "Person", "Building", 'Animal', 'Cartoons', 'Certificate', "Electronic", "Screenshot",
                  'BankCard', 'Mountain', 'Sea', "Bill", "Selfie", 'Night', 'Aircraft', 'Flower', "Child", "Ship"]  # key表示每个分类头的名称，value表示该分类头的类别数
device = torch.device("cuda:0")  

# 数据增强参数
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.513, 0.511, 0.505], [0.316, 0.307, 0.327]),
])

def get_cls(res):
    res_list = []
    for one in res:
        if res[one][0][0] > res[one][0][1]:
            res_list.append(0)
        else:
            res_list.append(1)
    return res_list

model = torch.load("C:/Users\hua\Desktop\code\checkpoint-15\\checkpoint-006-loss-1.51.pth", map_location='cpu',weights_only=False)
model.eval()

# 定义要预测的目录
image_dir = "C:/Users\hua\Desktop\code\\ablum\\test\ship"

# 遍历目录中的所有图片并进行预测
for fname in os.listdir(image_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(image_dir, fname)
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        image = torch.unsqueeze(image, dim=0)
        res = model(image)
        one_hot_res = get_cls(res)
        #print(f"Image: {fname}")
        #print("Predicted classes:",one_hot_res)
        for i in range(len(one_hot_res)):
            if one_hot_res[19] == 0:
                print(f"Image: {fname}")
                #print(class_names[i])
        #print("\n")