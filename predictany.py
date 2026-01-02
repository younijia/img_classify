import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from model import Wide_resnet50_2_ML_Dynamic

class_names = ['Vehicle', 'Sky', 'Food', "Person", "Building", 'Animal', 'Cartoons', 'Certificate', "Electronic", "Screenshot",
                  'BankCard', 'Mountain', 'Sea', "Bill", "Selfie", 'Night', 'Aircraft', 'Flower', "Child", "Ship"]
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

# 加载模型
model = torch.load(r"/home/yuanhua/xiangce/checkpoints-15/2025-03-03-18-07/checkpoint-019-loss-1.97.pth", map_location='cpu')
model.eval()

# 图片目录路径
img_dir = "/home/yuanhua/xiangce/album/test/Animal"
output = []

# 遍历目录中的所有图片文件
for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    
    # 只处理图片文件，忽略其他文件
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        try:
            # 打开并转换图像
            image = Image.open(img_path).convert('RGB')
            image = transform(image)
            image = torch.unsqueeze(image, dim=0)
            
            # 进行预测
            res = model(image)
            one_hot_res = get_cls(res)
            
            # 输出结果
            predicted_classes = [class_names[i] for i in range(len(one_hot_res)) if one_hot_res[i] == 1]
            output.append((img_name, predicted_classes))
            # 检测是否包含 'ship' 类
            if 'Animal'  in predicted_classes :
                print(f"Image in 'Animal' class: {img_name}")
                with open('Animal.txt', 'a') as f:
                    f.write(f"{img_name}\n")
                
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")

# # 输出所有图片的预测结果
# for img_name, predicted_classes in output:
#     print(f"Image: {img_name} -> Predicted Classes: {', '.join(predicted_classes)}")

