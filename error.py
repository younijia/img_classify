import os
from PIL import Image
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

def test_predictany():
    with open('album/onehot_test.txt', 'r') as f:
        all_labels = f.readlines()

    incorrect_child_classifications = []
    incorrect_ship_classifications = []
    incorrect_mountain_classifications = []
    incorrect_electronic_classifications = []
    incorrect_flower_classifications = []
    incorrect_animal_classifications = []

    for one in all_labels:
        img_name, label = one.strip().split("\t")
        label_list = [int(i) for i in label.split(",")]

        img_path = "album/img/" + img_name
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue

        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        image = torch.unsqueeze(image, dim=0)
        res = model(image)
        one_hot_res = get_cls(res)

        predicted_classes = [class_names[i] for i in range(len(one_hot_res)) if one_hot_res[i] == 1]

        # Check if the image is incorrectly classified as 'Child'
        if 'Ship' in predicted_classes and label_list[class_names.index('Ship')] == 0:
            incorrect_ship_classifications.append(img_name)
            print(f"Incorrectly classified as 'Ship': {img_name}")
        if 'Child'  in predicted_classes and label_list[class_names.index('Child')] == 0:
            incorrect_child_classifications.append(img_name)
            print(f"Incorrectly classified as 'Child': {img_name}")
        if 'Mountain' in predicted_classes and label_list[class_names.index('Mountain')] == 0:
            incorrect_mountain_classifications.append(img_name)
            print(f"Incorrectly classified as 'Mountain': {img_name}")
        if 'Electronic' in predicted_classes and label_list[class_names.index('Electronic')] == 0:
            incorrect_electronic_classifications.append(img_name)
            print(f"Incorrectly classified as 'Electronic': {img_name}")
        if 'Flower' in predicted_classes and label_list[class_names.index('Flower')] == 0:
            incorrect_flower_classifications.append(img_name)
            print(f"Incorrectly classified as 'Flower': {img_name}")
        if 'Animal' in predicted_classes and label_list[class_names.index('Animal')] == 0:
            incorrect_animal_classifications.append(img_name)
            print(f"Incorrectly classified as 'Animal': {img_name}")

    with open('incorrect_child_classifications.txt', 'w') as f:
        for img_name in incorrect_child_classifications:
            f.write(f"{img_name}\n")
    with open('incorrect_ship_classifications.txt', 'w') as f:
        for img_name in incorrect_ship_classifications:
            f.write(f"{img_name}\n")
    with open('incorrect_mountain_classifications.txt', 'w') as f:
        for img_name in incorrect_mountain_classifications:
            f.write(f"{img_name}\n")
    with open('incorrect_electronic_classifications.txt', 'w') as f:
        for img_name in incorrect_electronic_classifications:
            f.write(f"{img_name}\n")
    with open('incorrect_flower_classifications.txt', 'w') as f:
        for img_name in incorrect_flower_classifications:
            f.write(f"{img_name}\n")
    with open('incorrect_animal_classifications.txt', 'w') as f:
        for img_name in incorrect_animal_classifications:
            f.write(f"{img_name}\n")

if __name__ == "__main__":
    test_predictany()