import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from tqdm.auto import tqdm


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def validate_Dynamic(model, dataloader, logger, iteration, device, checkpoint=None, label_names=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        accuracies = {name: 0 for name in label_names}

        for i, batch in enumerate(tqdm(dataloader, desc='Validating')):
            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()

            batch_accuracies = calculate_metrics_Dynamic(output, target_labels, label_names)
            for name, batch_accuracy in zip(label_names, batch_accuracies.values()):
                accuracies[name] += batch_accuracy

    n_samples = len(dataloader)
    avg_loss /= n_samples
    for name in label_names:
        accuracies[name] /= n_samples

    # print('-' * 72)
    # print(f"Validation loss: {avg_loss:.4f}", end=', ')
    # for name in label_names:
    #     print(f"{name}: {accuracies[name]:.4f}", end=', ')
    # print("\n")

    for name in label_names:
        logger.add_scalar(f'val_loss', avg_loss, iteration)
        logger.add_scalar(f'val_accuracy_{name}', accuracies[name], iteration)

    model.train()
    return avg_loss, {"accuracy_" + name: accuracies[name] for name in label_names}


def calculate_metrics_Dynamic(output, target, label_names):
    # 初始化准确率字典
    accuracies = {}

    # 捕获警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in label_names:
            predicted = output[name].cpu()
            gt = target[f'label_{name}'].cpu()

            # 将预测值和真实值转换为 numpy 数组
            predicted_labels = np.argmax(predicted.detach().numpy(), axis=1)
            gt_labels = np.argmax(gt.numpy(), axis=1)

            # 计算准确率
            accuracy = balanced_accuracy_score(gt_labels, predicted_labels)
            accuracies[f'accuracy_{name}'] = accuracy

    return accuracies