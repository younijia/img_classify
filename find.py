import os
import shutil

def find_and_copy_images(file_list_path, src_dir, dst_dir):
    # 如果目标目录不存在，则创建
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # 读取txt文件中每一行的图片文件名，忽略空行
    with open(file_list_path, 'r', encoding='utf-8') as f:
        image_names = [line.strip() for line in f if line.strip()]
    
    # 对于每个图片文件名，在大目录的所有子目录中查找
    for img_name in image_names:
        found = False
        for root, dirs, files in os.walk(src_dir):
            if img_name in files:
                src_file_path = os.path.join(root, img_name)
                dst_file_path = os.path.join(dst_dir, img_name)
                shutil.copy(src_file_path, dst_file_path)
                print(f"复制成功：{src_file_path} -> {dst_file_path}")
                found = True
                # 如果只需要复制找到的第一个匹配项，则找到后跳出循环
                break
        if not found:
            print(f"未找到文件：{img_name}")

if __name__ == '__main__':
    # 根据实际情况修改下面的路径
    file_list_path = r"/home/yuanhua/xiangce/错漏分类/incorrect_ship_classifications.txt"  # 存放图片文件名的txt文件
    src_dir = r"/home/yuanhua/xiangce/album/test"  # 大目录路径，包含多个子目录
    dst_dir = './ship_error'  # 目标目录，复制的图片将存放到这里
    
    find_and_copy_images(file_list_path, src_dir, dst_dir)
