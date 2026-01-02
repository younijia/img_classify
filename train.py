import argparse
import os
from datetime import datetime
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset import MultiLabelDataset_Dynamic
from model import Wide_resnet50_2_ML_Dynamic
from test import validate_Dynamic, calculate_metrics_Dynamic
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import torch.optim as optim
from tqdm.auto import tqdm
from torch.nn.utils import clip_grad_norm


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')


def checkpoint_save_Dynamic(model, name, epoch, total_loss, val_accuracy):
    keys = list(val_accuracy.keys())
    file_name = "checkpoint-{:03d}-loss-{:.2f}".format(epoch, total_loss)
    #file_name = "checkpoint.pth"
    f = os.path.join(name, file_name)
    torch.save(model, f)
    print('Saved checkpoint:', f)


def make_label_info(class_info):
    label_info = {}
    sum = 0
    for key in class_info.keys():
        sum = sum + class_info[key]
        label_info["label_" + key] = sum
    return label_info


def make_label_names(class_info):
    return list(class_info.keys())


def make_accuracy_dict(class_info):
    return {"accuracy_" + key: 0 for key in class_info.keys()}


if __name__ == '__main__':
    train_txt_file = r'album/onehot_train_augmented.txt'  # 训练集路径
    val_txt_file = r'album/onehot_valid.txt'  # 验证集路径
    img_dir = '/home/yuanhua/xiangce/album/train'  # 图像路径
    img_dir_test = '/home/yuanhua/xiangce/album/img'  # 测试集图像路径
    class_info = {'Vehicle': 2, 'Sky': 2, 'Food': 2, "Person": 2, "Building": 2, 'Animal': 2, 'Cartoons': 2, 'Certificate': 2, "Electronic": 2, "Screenshot": 2,
                  'BankCard': 2, 'Mountain': 2, 'Sea': 2, "Bill": 2, "Selfie": 2, 'Night': 2, 'Aircraft': 2, 'Flower': 2, "Child": 2, "Ship": 2}  # key表示每个分类头的名称，value表示该分类头的类别数

    start_epoch = 1  # 从50个epoch后开始继续训练
    N_epochs = 20  # 训练总轮次
    batch_size = 32  # 批大小
    num_workers = 12  # 多线程加载数据
    device = torch.device("cuda:0")  # 使用哪块显卡

    label_info = make_label_info(class_info)
    label_names = make_label_names(class_info)

    transforms = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.3, hue=0.2),
        # transforms.RandomRotation(degrees=20),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.513, 0.511, 0.505], [0.316, 0.307, 0.327]),
    ])


    # 加载数据集
    train_dataset = MultiLabelDataset_Dynamic(train_txt_file, img_dir, transform=transforms, label_info=label_info)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataset = MultiLabelDataset_Dynamic(val_txt_file, img_dir_test, transform=transforms, label_info=label_info)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 加载模型
    model = Wide_resnet50_2_ML_Dynamic(img_bchw=(1, 3, 224, 224), class_info=class_info).to(device)

    # 设置优化器

    optimizer = torch.optim.AdamW(model.parameters(),lr=0.000005,weight_decay=0.1)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)   

    # 日志及模型保存路径
    logdir = os.path.join('./logs-15/', get_cur_time()) + "/"
    savedir = os.path.join('./checkpoints-15/', get_cur_time()) + "/"
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    n_train_samples = len(train_dataloader)
    print("Starting training ...")

    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []

    max_total_acc = 0  # 保存最优模型记录号
    for epoch in range(start_epoch, N_epochs + 1):
        start = time.time()
        total_loss = 0
        accuracy_dict = make_accuracy_dict(class_info)
        batch_index = 0
        for i, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}/{N_epochs}')):
            optimizer.zero_grad()

            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            
            # scaler = torch.cuda.amp.GradScaler()
            # # 前向传播使用autocast
            # with torch.cuda.amp.autocast():
            #     output = model(img.to(device))
            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)  # 必须先unscale
            # clip_grad_norm_(model.parameters(), max_norm=5)
            # scaler.step(optimizer)
            # scaler.update()

            output = model(img.to(device))
            loss_train, losses_train = model.get_loss(output, target_labels)
            total_loss += loss_train.item()
            temp_accuracy_dict = calculate_metrics_Dynamic(output, target_labels, label_names=label_names)

            for key in temp_accuracy_dict.keys():
                accuracy_dict[key] += temp_accuracy_dict[key]

            loss_train.backward()
            # # 在裁剪前打印梯度统计
            # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]), 2)
            # print(f"梯度范数：{total_norm.item():.2f}")
            #clip_grad_norm(model.parameters(), 3)
            optimizer.step()
        scheduler.step()
        print("当前学习率 lr: ", optimizer.param_groups[0]['lr'])
        end = time.time()

        print("train_epoch [{:4d}],loss: [{:.4f}]".format(epoch, total_loss / n_train_samples))
        print({key: round(accuracy_dict[key] / n_train_samples, 5) for key in accuracy_dict.keys()})
        avg_train_accuracy = round(float(sum([round(accuracy_dict[key] / n_train_samples , 5) for key in accuracy_dict.keys()])) / 20 , 5)
        print(" train accuracy: [", str(avg_train_accuracy), "]")
        train_acc_list.append(avg_train_accuracy)
        print("epoch:{:3d} use time: {:.4f}".format(epoch, end - start))

        logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)
        train_loss_list.append(round((total_loss / n_train_samples), 5))

        val_loss, val_accuracy = validate_Dynamic(model, val_dataloader, logger, epoch, device, label_names=label_names)

        print("<val_loss>: {:.4f}".format(val_loss), {key: round(val_accuracy[key], 3) for key in val_accuracy.keys()})
        avg_val_accuracy = round(float(sum(val_accuracy.values())) / 20 *100, 3)
        print("val accuracy: ", str(avg_val_accuracy))
        val_acc_list.append(avg_val_accuracy)
        val_loss_list.append(val_loss)

        if epoch == 1:
            checkpoint_save_Dynamic(model, savedir, epoch, val_loss, val_accuracy)
        elif epoch > 1 :
            if max_total_acc < sum(val_accuracy.values()):
                max_total_acc = sum(val_accuracy.values())
                checkpoint_save_Dynamic(model, savedir, epoch, val_loss, val_accuracy)
                print("保存模型成功")
            else:
                print("本次模型不如之前的模型，不保存")
        # elif epoch > 1 :
        #     if val_loss < min(val_loss_list):
        #         checkpoint_save_Dynamic(model, savedir, epoch, val_loss, val_accuracy)
        #         print("保存模型成功,loss: ", val_loss)
        #         print("val_loss_list: ", val_loss_list)
        #     else:
        #         print("本次模型不如之前的模型，不保存")

        # elif epoch == N_epochs:
        #     checkpoint_save_Dynamic(model, savedir, epoch, val_loss, val_accuracy)

    np.save(logdir + "train_loss.npy", np.array(train_loss_list))
    np.save(logdir + "train_acc.npy", np.array(train_acc_list))
    np.save(logdir + "val_loss.npy", np.array(val_loss_list))
    np.save(logdir + "val_acc.npy", np.array(val_acc_list))
    
