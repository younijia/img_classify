import os
from PIL import Image
from torchvision import transforms

# 数据增强参数
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize([0.513, 0.511, 0.505], [0.316, 0.307, 0.327]),
])

def augment_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            augmented_img = transform_train(img)
            augmented_img = transforms.ToPILImage()(augmented_img)
            augmented_img.save(os.path.join(output_dir, filename))

input_dir = '/home/yuanhua/xiangce/album/img'
output_dir = '/home/yuanhua/xiangce/album/new/output3'
augment_images(input_dir, output_dir)