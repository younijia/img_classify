import sys
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QGridLayout, QWidget, QLabel, QVBoxLayout, QScrollArea, QDialog
from PyQt5.QtGui import QPixmap
import numpy as np
from model import Wide_resnet50_2_ML_Dynamic

class_names = ['Vehicle', 'Sky', 'Food', "Person", "Building", 'Animal', 'Cartoons', 'Certificate', "Electronic", "Screenshot",
               'BankCard', 'Mountain', 'Sea', "Bill", "Selfie", 'Night', 'Aircraft', 'Flower', "Child", "Ship"]
device = torch.device("cuda:0")

# 数据增强参数
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class ImageItem:
    def __init__(self, filename):
        self.filename = filename
        self.is_classified = False
        self.predicted_classes = []

    def classify(self, predicted_classes):
        self.is_classified = True
        self.predicted_classes = predicted_classes

class Category:
    def __init__(self, name):
        self.name = name
        self.image_filenames = []

    def add_image(self, filename):
        self.image_filenames.append(filename)

def get_cls(res):
    res_list = []
    for one in res:
        if res[one][0][0] > res[one][0][1]:
            res_list.append(0)
        else:
            res_list.append(1)
    return res_list

# 加载模型
model = torch.load(r"/home/yuanhua/xiangce/checkpoints-15/2025-01-06-16-08/checkpoint-050-loss-12.08.pth", map_location='cpu')
model.eval()

# 图片目录路径
img_dir = "/home/yuanhua/xiangce/album/pic/"

# PyQt GUI类
class ImageClassificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('图像分类')
        self.setGeometry(100, 100, 1000, 600)

        self.categories = [Category(class_name) for class_name in class_names]
        self.images = {}

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # 分类按钮区域
        button_layout = QGridLayout()
        for i, category in enumerate(self.categories):
            button = QPushButton(category.name)
            button.clicked.connect(self.on_category_button_click)
            button_layout.addWidget(button, i // 5, i % 5)

        main_layout.addLayout(button_layout)

        # 分类按钮
        classify_button = QPushButton('开始分类')
        classify_button.clicked.connect(self.on_classify_button_click)
        main_layout.addWidget(classify_button)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def on_classify_button_click(self):
        # 对每张图像进行分类，并更新类别
        for img_name in os.listdir(img_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_item = ImageItem(img_name)
                self.images[img_name] = image_item

                img_path = os.path.join(img_dir, img_name)
                image = Image.open(img_path).convert('RGB')
                image = transform(image)
                image = torch.unsqueeze(image, dim=0)

                res = model(image)
                predicted_classes = get_cls(res)

                # 根据预测结果更新图像类别
                for i, predicted in enumerate(predicted_classes):
                    if predicted == 1:
                        self.categories[i].add_image(img_name)
                image_item.classify(predicted_classes)

        # 更新类别按钮中的图像数量显示
        for i, category in enumerate(self.categories):
            button = self.findChild(QPushButton, category.name)
            button.setText(f"{category.name} ({len(category.image_filenames)}张)")

    def on_category_button_click(self):
        # 显示所选类别的所有图像
        sender = self.sender()
        category_name = sender.text()

        category = next(cat for cat in self.categories if cat.name == category_name)
        self.show_images_in_category(category)

    def show_images_in_category(self, category):
        # 在新窗口显示该类别下的图像缩略图
        dialog = QDialog(self)
        layout = QVBoxLayout(dialog)

        scroll_area = QScrollArea(dialog)
        image_layout = QGridLayout(scroll_area)
        for i, img_name in enumerate(category.image_filenames):
            img_path = os.path.join(img_dir, img_name)
            pixmap = QPixmap(img_path).scaled(100, 100)  # 缩略图大小
            label = QLabel()
            label.setPixmap(pixmap)
            image_layout.addWidget(label, i // 5, i % 5)

        dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageClassificationApp()
    window.show()
    sys.exit(app.exec_())
