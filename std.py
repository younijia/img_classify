import os
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

def calculate_mean_std(image_dir):
    # 获取所有图片路径
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('jpg', 'png'))]
    
    # 初始化累加变量
    pixel_sum = np.zeros(3)  # 用于累加每个通道的像素值
    pixel_sq_sum = np.zeros(3)  # 用于累加每个通道的像素平方值
    num_pixels = 0  # 累加所有像素点数

    for image_path in image_paths:
        img = Image.open(image_path).convert('RGB')  # 打开图片并确保为RGB格式
        img_tensor = ToTensor()(img)  # 转换为Tensor，值范围 [0, 1]

        # 统计每张图片的像素均值和平方均值
        pixel_sum += img_tensor.sum(dim=(1, 2)).numpy()
        pixel_sq_sum += (img_tensor ** 2).sum(dim=(1, 2)).numpy()
        num_pixels += img_tensor.shape[1] * img_tensor.shape[2]  # 累加像素点数

    # 计算全局均值和标准差
    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sq_sum / num_pixels - mean ** 2)

    return mean.tolist(), std.tolist()

# 数据集目录
image_dir = "album/img"
mean, std = calculate_mean_std(image_dir)
print("Mean:", mean)
print("Std:", std)