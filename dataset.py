import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import torch


class MultiLabelDataset_Dynamic(Dataset):
    def __init__(self, txt_file, img_dir, transform=None, label_info=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []

        # 初始化标签列表
        self.labels = {name: [] for name in label_info.keys()}

        with open(txt_file, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                img_name = parts[0]
                labels = np.array([int(x) for x in parts[1].split(',')])

                # 动态分配标签
                start_idx = 0
                for name, end_idx in label_info.items():
                    end_idx = int(end_idx / 2)
                    # self.labels[name].append(np.squeeze(int(labels[start_idx:end_idx])))
                    if int(labels[start_idx:end_idx]) == 0:
                        self.labels[name].append(np.array([1, 0]))
                    else:
                        self.labels[name].append(np.array([0, 1]))
                    start_idx = end_idx
                self.img_labels.append((img_name, labels))

        # print(self.labels)
        # print(self.img_labels)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, _ = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 构建标签字典
        dict_data = {
            'img': image,
            'labels': {
                name: torch.tensor(self.labels[name][idx], dtype=torch.float)
                for name in self.labels.keys()
            }
        }

        return dict_data

    # one_hot = torch.zeros(np.array(batch_size, num_class, device=torch.device('cuda:0')).scatter_(1, label, 1)
