import os

def check_label_file_consistency(label_file, image_dir):
    # 读取标签文件中的文件名
    with open(label_file, 'r') as f:
        label_lines = f.readlines()
    
    label_filenames = set()
    for line in label_lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            filename, _ = parts
            label_filenames.add(filename)
    
    # 获取目录中的文件名
    image_filenames = set(os.listdir(image_dir))
    
    # 找出差异
    missing_in_labels = image_filenames - label_filenames
    missing_in_directory = label_filenames - image_filenames
    
    print(f"标签文件中的总数量: {len(label_filenames)}")
    print(f"目录中的总数量: {len(image_filenames)}")
    
    if missing_in_labels:
        print("以下文件在目录中存在但未出现在标签文件中:")
        for filename in missing_in_labels:
            print(filename)
    
    if missing_in_directory:
        print("以下文件在标签文件中存在但未出现在目录中:")
        for filename in missing_in_directory:
            print(filename)
    
    if not missing_in_labels and not missing_in_directory:
        print("文件匹配完全一致，无差异。")

    print(f"标签文件中的总数量: {len(label_filenames)}")
    print(f"目录中的总数量: {len(image_filenames)}")

# 使用示例
check_label_file_consistency(r"C:\\Users\\hua\Desktop\\code\\ablum\\onehot_train_augmented.txt", r"C:\\Users\\hua\Desktop\\code\\ablum\\train")