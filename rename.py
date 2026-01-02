import os

def rename_images(folder_path, prefix="aug", start_index=1, file_exts=(".jpg", ".png")):
    """
    批量重命名文件夹中的图像文件

    :param folder_path: 图像文件所在的文件夹路径
    :param prefix: 重命名的前缀，默认 "aug"
    :param start_index: 起始编号，默认从 1 开始
    :param file_exts: 需要重命名的文件类型，默认支持 .jpg 和 .png
    """
    if not os.path.exists(folder_path):
        print("目录不存在！")
        return
    
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(file_exts)]
    images.sort()  # 按文件名排序，保证顺序一致

    for idx, image in enumerate(images, start=start_index):
        old_path = os.path.join(folder_path, image)
        new_name = f"{prefix}_{idx}{os.path.splitext(image)[1]}"  # 保持原始后缀
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)
        print(f"重命名: {image} -> {new_name}")

# 示例使用
rename_images("path/to/augmented_images", prefix="dataset_aug", start_index=1000)
