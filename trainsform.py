import os
from PIL import Image
from torchvision import transforms

# 数据增强参数
transform_train = transforms.Compose([
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.ColorJitter(brightness=0.3, contrast=0.2),
    #transforms.RandomRotation(degrees=30),
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
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_augmented{ext}"
            augmented_img = augmented_img.convert("RGB")  # 去掉透明通道
            augmented_img.save(os.path.join(output_dir, new_filename))

input_dir = r"C:\\Users\\hua\Desktop\\code\\ablum\\img"
output_dir = r"C:\\Users\\hua\Desktop\\code\\ablum\\img_first"
augment_images(input_dir, output_dir)