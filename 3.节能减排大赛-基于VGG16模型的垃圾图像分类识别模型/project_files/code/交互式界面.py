import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import VGG16_Weights
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                             QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont, QImage
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持
try:
    from matplotlib import font_manager

    font_path = font_manager.findfont(font_manager.FontProperties(family='SimHei'))
    chinese_font = ImageFont.truetype(font_path, 20)
except:
    chinese_font = None
    print("警告: 未找到中文字体，可能影响中文显示")


# 垃圾分类模型类
class RubbishClassifier:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.classes = ['battery','food','glass','medical trash','metal','paper','plastic']  # 示例类别，请根据您的实际类别修改

        # 初始化模型
        self.model = self._create_model()

        # 加载预训练权重
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"成功加载模型权重：{model_path}")
        except Exception as e:
            raise ValueError(f"加载模型权重失败：{str(e)}")

        self.model.to(self.device)
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _create_model(self):
        """创建与训练时相同的模型结构"""
        vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # 冻结特征提取层
        for param in vgg16.features.parameters():
            param.requires_grad = False

        # 修改分类器为7类输出（请根据您的实际类别数量修改）
        vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(self.classes))  # 类别数量
        )

        return vgg16

    def predict(self, image_path):
        """对单张图像进行预测"""
        try:
            # 读取图像
            image = Image.open(image_path).convert('RGB')
            original_image = image.copy()

            # 预处理
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            # 获取结果
            predicted_class = self.classes[predicted_idx.item()]
            confidence_value = confidence.item() * 100

            # 在图像上添加预测结果
            draw = ImageDraw.Draw(original_image)
            text = f"{predicted_class}: {confidence_value:.2f}%"

            if chinese_font:
                # 添加中文类别名称（如果有对应关系）
                chinese_names = {
                    'battery': '电池',
                    'food': '厨余垃圾',
                    'glass': '金属',
                    'medical trash': '其他垃圾',
                    'metal': '金属',
                    'paper': '纸',
                    'plastic': '塑料',
                }
                chinese_text = f"{chinese_names.get(predicted_class, predicted_class)}: {confidence_value:.2f}%"
                draw.text((10, 10), chinese_text, fill=(255, 0, 0), font=chinese_font)
            else:
                draw.text((10, 10), text, fill=(255, 0, 0))

            return original_image, predicted_class, confidence_value

        except Exception as e:
            raise ValueError(f"预测失败：{str(e)}")


# 主界面类
class RubbishClassificationApp(QMainWindow):
    def __init__(self, model_path):
        super().__init__()
        self.classifier = RubbishClassifier(model_path)
        self.current_image_path = None
        self.initUI()

    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle('垃圾分类识别系统')
        self.setGeometry(100, 100, 1000, 700)

        # 中央部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 标题
        title_label = QLabel('垃圾分类图像识别系统')
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setText("请上传图片进行识别")
        self.image_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.image_label)

        # 按钮区域
        button_layout = QHBoxLayout()

        self.upload_btn = QPushButton('上传图片')
        self.upload_btn.clicked.connect(self.upload_image)
        button_layout.addWidget(self.upload_btn)

        self.predict_btn = QPushButton('开始识别')
        self.predict_btn.clicked.connect(self.predict_image)
        self.predict_btn.setEnabled(False)
        button_layout.addWidget(self.predict_btn)

        self.clear_btn = QPushButton('清除结果')
        self.clear_btn.clicked.connect(self.clear_result)
        button_layout.addWidget(self.clear_btn)

        self.quit_btn = QPushButton('退出程序')
        self.quit_btn.clicked.connect(self.close)
        button_layout.addWidget(self.quit_btn)

        layout.addLayout(button_layout)

        # 结果显示区域
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(100)
        layout.addWidget(self.result_text)

        # 状态栏
        self.statusBar().showMessage('就绪')

    def upload_image(self):
        """上传图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择图片', '',
            '图像文件 (*.png *.jpg *.jpeg *.bmp *.gif *.webp)')

        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(file_path)

            # 调整图像大小以适应显示区域
            scaled_pixmap = pixmap.scaled(
                400, 400,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.image_label.setPixmap(scaled_pixmap)
            self.predict_btn.setEnabled(True)
            self.result_text.clear()
            self.statusBar().showMessage(f'已加载图片: {os.path.basename(file_path)}')

    def predict_image(self):
        """预测图片"""
        if not self.current_image_path:
            QMessageBox.warning(self, '警告', '请先上传图片!')
            return

        try:
            # 显示等待提示
            self.statusBar().showMessage('正在识别中...')
            QApplication.processEvents()  # 更新UI

            # 进行预测
            result_image, predicted_class, confidence = self.classifier.predict(self.current_image_path)

            # 转换PIL图像为QPixmap
            result_image = result_image.convert("RGB")
            data = result_image.tobytes("raw", "RGB")
            qim = QImage(data, result_image.size[0], result_image.size[1], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qim)

            # 调整图像大小以适应显示区域
            scaled_pixmap = pixmap.scaled(
                400, 400,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.image_label.setPixmap(scaled_pixmap)

            # 显示结果
            result_message = f"识别结果: {predicted_class}\n置信度: {confidence:.2f}%"
            self.result_text.setPlainText(result_message)
            self.statusBar().showMessage('识别完成')

        except Exception as e:
            QMessageBox.critical(self, '错误', f'识别过程中发生错误: {str(e)}')
            self.statusBar().showMessage('识别失败')

    def clear_result(self):
        """清除结果"""
        self.image_label.clear()
        self.image_label.setText("请上传图片进行识别")
        self.result_text.clear()
        self.current_image_path = None
        self.predict_btn.setEnabled(False)
        self.statusBar().showMessage('已清除结果')

    def closeEvent(self, event):
        """关闭程序前的确认"""
        reply = QMessageBox.question(
            self, '确认退出',
            '确定要退出程序吗?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


# 主函数
def main():
    # 请修改为您的模型权重文件路径
    model_path = "vgg16_rubbish_model.pth"

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请修改model_path变量为正确的模型文件路径")
        return

    app = QApplication(sys.argv)
    window = RubbishClassificationApp(model_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()