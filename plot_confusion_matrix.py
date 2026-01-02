import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import os

class_names = ['Vehicle', 'Sky', 'Food', "Person", "Building", 'Animal', 'Cartoons', 'Certificate', "Electronic", "Screenshot",
                  'BankCard', 'Mountain', 'Sea', "Bill", "Selfie", 'Night', 'Aircraft', 'Flower', "Child", "Ship"]

# 所有类别
classes  = ['No', 'Yes']
# 绘制混淆矩阵函数
def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的数量值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        plt.text(x_val, y_val, "%d" % (c,), color='black', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Set2)
    plt.title(title, fontsize=15)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90, fontsize=15)
    plt.yticks(xlocations, classes, fontsize=15)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predict label', fontsize=15)

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    output_dir = '/home/yuanhua/xiangce/confusion0306'
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, savename)
    plt.savefig(outpath, format='png')
    # plt.show()

all_predicts = np.load("predict-0306.npy")
all_labels = np.load("label-0306.npy")
for index in range(0, 20):
    name = class_names[index]
    lb = np.squeeze(all_labels[:, index])
    pd = np.squeeze(all_predicts[:, index])
    mat = confusion_matrix(lb, pd)
    plot_confusion_matrix(mat, name + '_confusion_matrix.png', title=name + ' confusion matrix')

    # 计算并打印准确率、精确率、召回率和F1分数
    accuracy = accuracy_score(lb, pd)
    precision = precision_score(lb, pd, average='binary')
    recall = recall_score(lb, pd, average='binary')
    f1 = f1_score(lb, pd, average='binary')
    print(f"{name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")