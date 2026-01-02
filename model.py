import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import timm

import torch
import torch.nn as nn


# swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# 多任务损失函数
class MultiTaskLossModule_Dynamic(nn.Module):
    def __init__(self, class_info):
        super(MultiTaskLossModule_Dynamic, self).__init__()
        self.tasks = {task: "label_" + task for task in class_info.keys()}

    def _calculate_loss(self, output_key, label_key, net_output, ground_truth):
        """计算单个任务的损失"""
        return F.cross_entropy(net_output[output_key], ground_truth[label_key])

    def get_loss(self, net_output, ground_truth):
        # 定义任务及其对应的输出和标签键

        total_loss = 0
        individual_losses = {}

        for task, label_key in self.tasks.items():
            output_key = task  # 假设输出键与任务名称相同
            loss = self._calculate_loss(output_key, label_key, net_output, ground_truth)
            total_loss += loss
            individual_losses[task] = loss

        return total_loss, individual_losses

# ResNet50
class Wide_resnet50_2_ML_Dynamic(MultiTaskLossModule_Dynamic):
    def __init__(self, img_bchw, class_info):
        super(Wide_resnet50_2_ML_Dynamic, self).__init__(class_info)

        # 加载预训练的WideResNet50_2模型
        self.wide_resnet = models.wide_resnet50_2(pretrained=True)

        # 替换原始的分类层
        num_ftrs = self.wide_resnet.fc.in_features
        self.wide_resnet.fc = nn.Identity()

        # 添加Batch Normalization和Dropout层
        self.bn = nn.BatchNorm1d(num_ftrs)
        self.dropout = nn.Dropout(0.5)

        # 动态创建分类头
        self.classifiers = nn.ModuleDict()
        for name, n_classes in class_info.items():
            self.classifiers[name] = nn.Linear(num_ftrs, n_classes)

        # 添加Swish激活函数
        self.swish = Swish()
    def forward(self, x):
        # 使用WideResNet50_2的特征提取部分
        x = self.wide_resnet(x)

        # 应用Batch Normalization和Dropout
        x = self.bn(x)

        # 应用Swish激活函数
        x = self.swish(x)
        x = self.dropout(x)



        # 通过各自的分类头
        outputs = {}
        for name, classifier in self.classifiers.items():
            outputs[name] = classifier(x)

        return outputs
