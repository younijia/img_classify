import sys
import os
import threading
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize,QObject
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QGridLayout, QLabel, QVBoxLayout,
    QScrollArea, QProgressDialog, QMainWindow, QFileDialog
)
import torch
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torchvision.transforms as transforms

# 图片类，提供判断是否被分类的属性
class MyImage:
    def __init__(self, file_path):
        self.file_path = file_path
        self.is_classified = False  # 是否被分类

# 文件监控类
class FileMonitor(QObject, FileSystemEventHandler): 
    image_added = pyqtSignal()  # 发出信号通知新图片添加
    image_deleted = pyqtSignal()  # 发出信号通知图片被删除
    def __init__(self, all_images):
        super().__init__()  # 调用 QObject 的初始化方法
        self.all_images = all_images
        self.lock = threading.Lock()  # 添加锁

    def normalize_path(self, path):
        """规范化路径，确保比较一致性"""
        return os.path.normpath(os.path.abspath(path)).lower()

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            new_image = MyImage(event.src_path)
            with self.lock:  # 获取锁，保证线程安全
                self.all_images.append(new_image)
            print(f"检测到新图片: {event.src_path}")
            self.image_added.emit()  # 发出图片添加信号

    def on_deleted(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            with self.lock:  # 获取锁，保证线程安全
                self.all_images = [img for img in self.all_images if self.normalize_path(img.file_path) != self.normalize_path(event.src_path)]
            print(f"图片被删除: {event.src_path}")
            self.image_deleted.emit()  # 发出图片删除信号
# 文件监控线程类
class FileMonitorThread(QThread):
    def __init__(self, all_images, directory, main_window):
        super().__init__()
        self.all_images = all_images
        self.directory = directory
        self.monitor = FileMonitor(self.all_images)
        self.main_window = main_window  # 保存主窗口实例

    def run(self):
        self.monitor.image_added.connect(self.update_stats)  # 连接信号
        self.monitor.image_deleted.connect(self.update_stats)  # 连接信号
        self.observer = Observer()
        self.observer.schedule(self.monitor, self.directory, recursive=False)
        self.observer.start()
        self.observer.join()

    def stop(self):
        self.observer.stop()
        self.observer.join()

    def update_stats(self):
        if self.main_window:
            self.main_window.update_stats()  # 调用主窗口的 update_stats

# 20个类别数组，用来存储对应类别下的图片文件名
class_names = [
    'Vehicle', 'Sky', 'Food', 'Person', 'Building', 'Animal', 'Cartoons', 'Certificate',
    'Electronic', 'Screenshot', 'BankCard', 'Mountain', 'Sea', 'Bill', 'Selfie',
    'Night', 'Aircraft', 'Flower', 'Child', 'Ship'
]
category_images = {name: [] for name in class_names}

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_cls(res):
    res_list = []
    for one in res:
        if res[one][0][0] > res[one][0][1]:
            res_list.append(0)
        else:
            res_list.append(1)
    return res_list

# 分类线程类
class ClassificationThread(QThread):
    update_signal = pyqtSignal()

    def __init__(self, all_images, model, device):
        super().__init__()
        self.all_images = all_images
        self.model = model
        self.device = device
        self.lock = threading.Lock()  # 添加锁


    def run(self):
        for img in self.all_images:
            if not img.is_classified:
                try:
                    image_pil = Image.open(img.file_path).convert('RGB')
                    image_t = transform(image_pil).unsqueeze(0).to(self.device)
                    res = self.model(image_t)
                    one_hot_res = get_cls(res)  # 根据实际模型输出处理
                    img.is_classified = True
                    # 将图片加入对应类别
                    with self.lock:  # 获取锁，保证线程安全
                        for i, val in enumerate(one_hot_res):
                            if val == 1:
                                category_images[class_names[i]].append(img.file_path)
                except Exception as e:
                    print(f"Error processing image {img.file_path}: {e}")
        self.update_signal.emit()
# 主窗口类
class MainWindow(QMainWindow):
    def __init__(self, all_images, image_dir):
        super().__init__()
        self.setWindowTitle("图像多标签分类系统")
        self.setMinimumSize(800, 600)
        self.resize(1380, 720)
        self.all_images = all_images
        self.image_dir = image_dir
        self.file_monitor_thread = None
        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.stats_label = QLabel()
        self.layout.addWidget(self.stats_label)
        # 添加刷新按钮
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh_file_count)  # 连接刷新按钮的槽函数
        self.layout.addWidget(self.refresh_btn)

        self.start_btn = QPushButton("开始分类")
        self.start_btn.clicked.connect(self.classify_images)
        self.layout.addWidget(self.start_btn)

        # 添加退出按钮
        self.exit_btn = QPushButton("退出系统")
        self.exit_btn.clicked.connect(self.exit_system)
        self.layout.addWidget(self.exit_btn)

        self.grid_layout = QGridLayout()
        row = col = 0
        for name in class_names:
            btn = QPushButton(f"{name} (0)")
            icon_path = f"icons/{name.lower()}.jpg"
            if os.path.exists(icon_path):
                btn.setIcon(QIcon(icon_path))
                btn.setIconSize(QSize(150, 150))  # 设置图标大小
            btn.clicked.connect(lambda _, n=name: self.show_category_images(n))
            btn.setStyleSheet("font: 16pt 'Arial'; font-weight: bold; color: blue;")
            self.grid_layout.addWidget(btn, row, col)
            col += 1
            if col == 5:
                col = 0
                row += 1

        self.layout.addLayout(self.grid_layout)
        self.update_stats()
         # 设置按钮背景色
        self.refresh_btn.setStyleSheet("background-color: lightblue;")
        self.start_btn.setStyleSheet("background-color: lightgreen;")
        self.exit_btn.setStyleSheet("background-color: lightcoral;")

        # 设置主页面背景色
        self.central_widget.setStyleSheet("background-color: lightgray;")

    def exit_system(self):
        """退出系统"""
        self.close()

    def refresh_file_count(self):
        """刷新当前目录下的图片文件数量，并更新显示"""
        existing_files = {img.file_path for img in self.all_images}
        current_files = {os.path.join(self.image_dir, fname) for fname in os.listdir(self.image_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}

        # 找出新增和删除的文件
        new_files = current_files - existing_files
        deleted_files = existing_files - current_files

        # 更新 all_images 列表
        self.all_images = [img for img in self.all_images if img.file_path not in deleted_files]
        for file_path in new_files:
            self.all_images.append(MyImage(file_path))

        # 更新类别图片数量
        for name in class_names:
            category_images[name] = [img for img in category_images[name] if img in current_files]

        self.update_stats()  # 更新统计信息
        # 启动文件监控线程，并传递主窗口实例
        # self.file_monitor_thread = FileMonitorThread(self.all_images, self.image_dir, self)
        # self.file_monitor_thread.start()
        # 启动文件监控线程，并传递主窗口实例
        if self.file_monitor_thread is None:
            self.file_monitor_thread = FileMonitorThread(self.all_images, self.image_dir, self)
            self.file_monitor_thread.start()

    def classify_images(self):
        # # 清空所有类别下的图片列表
        # for name in class_names:
        #     category_images[name].clear()

        # 更新按钮文本
        index = 0
        for name in class_names:
            btn = self.grid_layout.itemAt(index).widget()
            btn.setText(f"{name} (0)")
            index += 1
        self.progress_dialog = QProgressDialog("正在进行分类，请稍后...", None, 0, 0, self)
        self.progress_dialog.setWindowTitle("进行中，请稍后")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()

        self.classification_thread = ClassificationThread(self.all_images, model, device)
        self.classification_thread.update_signal.connect(self.on_classification_complete)
        self.classification_thread.start()

    def on_classification_complete(self):
        index = 0
        for name in class_names:
            btn = self.grid_layout.itemAt(index).widget()
            btn.setText(f"{name} ({len(category_images.get(name, []))})")
            index += 1
        self.update_stats()
        self.progress_dialog.close()

    def show_category_images(self, category_name):
        self.hide()
        self.thumb_view = ThumbnailView(category_name, self)
        self.thumb_view.show()

    def remove_image(self, img_path):
        self.all_images = [img for img in self.all_images if img.file_path != img_path]
        self.update_stats()

    def update_stats(self):
        total_images = len(self.all_images)
        print(total_images)
        classified_images = sum(1 for img in self.all_images if img.is_classified)
        unclassified_images = total_images - classified_images
        self.stats_label.setText(f"总图片数: {total_images}  已分类: {classified_images}  未分类: {unclassified_images}")

        # 更新类别按钮文本
        index = 0
        for name in class_names:
            btn = self.grid_layout.itemAt(index).widget()
            btn.setText(f"{name} ({len(category_images.get(name, []))})")
            index += 1

    # def closeEvent(self, event):
    #     self.file_monitor_thread.stop()
    #     event.accept()
    def closeEvent(self, event):
        if self.file_monitor_thread:
            self.file_monitor_thread.stop()
        event.accept()
# 缩略图视图类
class ThumbnailView(QWidget):
    def __init__(self, category_name, parent_window):
        super().__init__()
        self.setWindowTitle(category_name)
        self.setMinimumSize(800, 600)  # 设置窗口最小尺寸
        self.resize(1024, 768)  # 设置窗口默认尺寸
        self.category_name = category_name
        self.parent_window = parent_window
        self.layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_content)  # 使用 QGridLayout
        self.scroll_area.setWidget(self.scroll_content)
        self.layout.addWidget(self.scroll_area)
        self.back_btn = QPushButton("返回")
        self.back_btn.clicked.connect(self.back_to_main)
        self.layout.addWidget(self.back_btn)
        self.setLayout(self.layout)
        self.fullscreen_label = QLabel()
        self.fullscreen_label.setAlignment(Qt.AlignCenter)
        self.fullscreen_label.hide()
        self.layout.addWidget(self.fullscreen_label)
        self.update_images()

    def update_images(self):
        self.clear_layout(self.scroll_layout)
        row = col = 0
        for img_path in category_images[self.category_name]:
            label = QLabel()
            pixmap = QPixmap(img_path)
            label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio))
            label.mousePressEvent = lambda event, path=img_path: self.toggle_fullscreen_image(path)
            self.scroll_layout.addWidget(label, row, col)
            delete_btn = QPushButton("删除")
            delete_btn.clicked.connect(lambda _, path=img_path: self.delete_image(path))
            self.scroll_layout.addWidget(delete_btn, row, col + 1)
            col += 2
            if col >= 6:  # 每行最多显示3个图片和删除按钮
                col = 0
                row += 1

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def toggle_fullscreen_image(self, img_path):
        if self.fullscreen_label.isVisible():
            self.fullscreen_label.hide()
            self.scroll_area.show()
            self.back_btn.setText("返回")
            self.back_btn.clicked.disconnect()
            self.back_btn.clicked.connect(self.back_to_main)
        else:
            pixmap = QPixmap(img_path)
            self.fullscreen_label.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio))
            self.fullscreen_label.show()
            self.scroll_area.hide()
            self.back_btn.setText("返回缩略图")
            self.back_btn.clicked.disconnect()
            self.back_btn.clicked.connect(self.back_to_thumbnail)

    def delete_image(self, img_path):
        category_images[self.category_name].remove(img_path)
        self.parent_window.remove_image(img_path)
        # 删除文件系统中的图片
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"Deleted file: {img_path}")
        else:
            print(f"File not found: {img_path}")
        self.update_images()
        self.parent_window.update_stats()

    def back_to_thumbnail(self):
        self.fullscreen_label.hide()
        self.scroll_area.show()
        self.back_btn.setText("返回")
        self.back_btn.clicked.disconnect()
        self.back_btn.clicked.connect(self.back_to_main)

    def back_to_main(self):
        self.hide()
        self.parent_window.show()
def main():
    app = QApplication(sys.argv)
    image_dir = QFileDialog.getExistingDirectory(None, "选择图片文件夹", "")
    if not image_dir:
        return

    all_images = []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_path = os.path.join(image_dir, fname)
            all_images.append(MyImage(file_path))

    main_window = MainWindow(all_images, image_dir)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    # 深度学习模型加载示例
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "C:/Users/hua/Desktop/code/checkpoint-15/checkpoint.pth"
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    main()