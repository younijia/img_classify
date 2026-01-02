import numpy as np
import matplotlib.pyplot as plt

# 假设你已经知道日志文件所在路径
logdir = './logs-15/2025-02-25-13-42/'  # 这里填入你保存数据的路径

# 加载训练和验证数据
train_loss = np.load(logdir + "train_loss.npy")
train_acc = np.load(logdir + "train_acc.npy")
val_loss = np.load(logdir + "val_loss.npy")
val_acc = np.load(logdir + "val_acc.npy")

# 获取训练轮数
epochs = np.arange(1, len(train_loss) + 1)

# 绘制训练和验证损失曲线
plt.figure(figsize=(12, 6))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss', color='blue')
plt.plot(epochs, val_loss, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()

# 准确度曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Train Accuracy', color='blue')
plt.plot(epochs, val_acc, label='Validation Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()
