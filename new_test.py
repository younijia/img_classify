import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from model import Wide_resnet50_2_ML_Dynamic
import time

class_info = {'Vehicle': 2, 'Sky': 2, 'Food': 2, "Person": 2, "Building": 2, 'Animal': 2, 'Cartoons': 2, 'Certificate': 2, "Electronic": 2, "Screenshot": 2,
                  'BankCard': 2, 'Mountain': 2, 'Sea': 2, "Bill": 2, "Selfie": 2, 'Night': 2, 'Aircraft': 2, 'Flower': 2, "Child": 2, "Ship": 2}  # key表示每个分类头的名称，value表示该分类头的类别数
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

def compare_label(predict_label, label):
    true_num = 0
    error_num = 0
    for i in range(20):
        if predict_label[i] == label[i]:
            true_num += 1
        else:
            error_num += 1
    return true_num, error_num
# model = Wide_resnet50_2_ML_Dynamic(img_bchw=(1, 3, 224, 224), class_info=class_info).to(device)
# model.load_state_dict(torch.load(r"D:\MyProjects\xiangce\checkpoints-15\2025-01-03-10-11\checkpoint-074-loss-7.48.pth", map_location='cpu'))
model = torch.load(r"C:\\Users\\hua\Desktop\\code\\checkpoint-15\\checkpoint-019-loss-1.97.pth", map_location='cpu',weights_only=False)
model.eval()

with open(r"C:\\Users\\hua\Desktop\\code\\ablum\\onehot_test.txt", "r") as f:
    all_labels = f.readlines()
    f.close()
print(len(all_labels))

all_predicts = []
all_trues = []
all_true = 0
all_error = 0
total_time = 0
for index in range(len(all_labels)):
    print("*"*50 + str(index+1) + "/" + str(len(all_labels)) + "*"*50)
    one = all_labels[index]
    img_name, label = one.strip().split("\t")
    label_list = [int(i) for i in label.split(",")]
    all_trues.append(label_list)

    img_path = r"C:\\Users\\hua\Desktop\\code\\ablum\\img\\" + img_name
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    image = torch.unsqueeze(image, dim=0)

    start_time = time.time()
    res = model(image)
    end_time = time.time()
    total_time += (end_time - start_time)
    one_hot_res = get_cls(res)
    all_predicts.append(one_hot_res)
    # one_true, one_error = compare_label(one_hot_res, label_list)
    # all_true += one_true
    # all_error += one_error

# print(all_true / (all_true + all_error))

average_time_per_image = total_time / len(all_labels)
print(f"Average time per image: {average_time_per_image} seconds")

all_predicts_np = np.array(all_predicts)
all_trues_np = np.array(all_trues)
np.save("predict0306.npy", all_predicts_np)
np.save("label0306.npy", all_trues_np)


# img_path = "album/img/000003.png"
# image = Image.open(img_path).convert('RGB')
# image = transform(image)
# image = torch.unsqueeze(image, dim=0)
# print(image.shape)
# res = model(image)
# print(res)
# one_hot_res = get_cls(res)
# print(one_hot_res)