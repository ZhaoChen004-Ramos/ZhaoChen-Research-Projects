import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import VGG16_Weights
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import matplotlib.pyplot as plt
import random
import warnings
plt.rcParams['font.family'] = ['SimHei']  # 设置全局字体为黑体，支持中文
plt.rcParams['axes.unicode_minus'] = False  # 可选：解决负号“-”显示为方块的问题
# 忽略不必要的警告
warnings.filterwarnings('ignore')


class RubbishDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []  # 存储所有有效图片路径
        self.labels = []  # 存储对应标签
        self.classes = []  # 存储类别名称

        # 获取所有类别文件夹
        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])

        # 遍历每个类别文件夹，收集图片路径和标签
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            # 遍历文件夹中的所有文件
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                # 检查文件是否为图片（支持多种格式）
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)

        print(f"成功加载数据集：共 {len(self.classes)} 类，{len(self.image_paths)} 张图片")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # 加载并转换图像
            image = Image.open(self.image_paths[idx]).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"加载图片出错 {self.image_paths[idx]}: {str(e)}")
            # 返回一个随机样本作为替代（避免程序中断）
            random_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(random_idx)


# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集并划分训练/测试集（8:2比例）
# 请修改为你的实际路径
dataset = RubbishDataset('C:/Users/HP/Desktop/Data/rubbish', transform=transform)  # 假设路径修改为此

# 确保数据集不为空
if len(dataset) == 0:
    raise ValueError("未找到任何图片，请检查数据集路径是否正确")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)  # 关闭多进程加载
test_loader = DataLoader(test_set, batch_size=16, num_workers=0)

# 加载预训练VGG16并修改分类器（输出7类）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 使用新的weights参数代替pretrained
vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

# 冻结特征提取层
for param in vgg16.features.parameters():
    param.requires_grad = False

# 修改分类器（输出7类）
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
    nn.Linear(256, 7)  # 7类垃圾
)
model = vgg16.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

# 模型训练
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 每10个batch打印一次进度
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}] 完成, 平均Loss: {epoch_loss:.4f}\n')

# 模型测试
model.eval()
total = 0
correct = 0
class_correct = [0] * 7
class_total = [0] * 7

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 计算每个类别的准确率
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# 打印总体准确率
accuracy = 100 * correct / total
print(f'测试集总体准确率: {accuracy:.2f}%')

# 打印每个类别的准确率
print("\n各类别准确率:")
for i in range(7):
    if class_total[i] > 0:
        print(
            f'类别 {dataset.classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})')

# 可视化测试集样本的预测结果
model.cpu()
idxs = random.sample(range(len(test_set)), min(10, len(test_set)))  # 确保不超过测试集大小
plt.figure(figsize=(15, 5))

# 反归一化（用于正确显示图像）
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

for i, idx in enumerate(idxs):
    img, label = test_set[idx]
    with torch.no_grad():
        output = model(img.unsqueeze(0))
        pred = torch.argmax(output, dim=1).item()

    # 反归一化后显示图像
    img_show = inv_normalize(img)
    plt.subplot(2, 5, i + 1)
    plt.imshow(img_show.permute(1, 2, 0).clamp(0, 1))  # 确保像素值在0-1范围内
    plt.title(f'预测: {dataset.classes[pred]}\n真实: {dataset.classes[label]}')
    plt.axis('off')

plt.tight_layout()
plt.show()
