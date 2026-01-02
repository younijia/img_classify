import os
import shutil

# 定义类别名称
categories = [
    "Vehicle", "Sky", "Food", "Person", "Building", "Animal", "Cartoons", "Certificate", "Electronic", "Screenshot",
    "BankCard", "Mountain", "Sea", "Bill", "Selfie", "Night", "Aircraft", "Flower", "Child", "Ship"
]

# 设置路径
image_dir = r"C:\\Users\\hua\Desktop\\code\\ablum\\img"  # 修改为图片所在目录
txt_file = r"C:\\Users\\hua\Desktop\\code\\ablum\\onehot_test.txt"  # 修改为txt文件路径
output_dir = r"C:\\Users\\hua\Desktop\\code\\ablum\\test"  # 修改为输出目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 创建类别目录
category_dirs = {category: os.path.join(output_dir, category) for category in categories}
for path in category_dirs.values():
    os.makedirs(path, exist_ok=True)

# 读取txt文件并分类图片
with open(txt_file, "r") as file:
    for line in file:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        image_name, labels = parts
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found!")
            continue
        
        label_list = list(map(int, labels.split(",")))
        for i, label in enumerate(label_list):
            if label == 1:
                shutil.copy(image_path, os.path.join(category_dirs[categories[i]], image_name))

print("Images have been categorized successfully!")
