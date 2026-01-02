import os

def augment_label_file(input_file):
    # 生成新的文件名
    name, ext = os.path.splitext(input_file)
    output_file = f"{name}_augmented{ext}"
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    augmented_lines = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            filename, labels = parts
            new_filename = f"{os.path.splitext(filename)[0]}_augmented{os.path.splitext(filename)[1]}"
            augmented_lines.append(f"{new_filename}\t{labels}\n")
    
    with open(output_file, 'w') as f:
        f.writelines(lines)  # 写入原始数据
        f.writelines(augmented_lines)  # 追加新数据
    
    print(f"文件已生成: {output_file}")

# 使用示例
augment_label_file(r"C:\\Users\\hua\Desktop\\code\\ablum\\onehot_train.txt")