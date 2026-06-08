# pix2pix 图像翻译实验

**学生：** 赵宸  
**学号：** 2302020530  
**班级：** 化学 235  
**实验日期：** 2025.7.27  
**实验平台：** Conda / PyCharm / 终端

---

## 目录

- [项目简介](#项目简介)
- [环境配置](#环境配置)
- [实验目的](#实验目的)
- [Pix2Pix 原理简介](#pix2pix-原理简介)
- [数据集说明](#数据集说明)
- [实验步骤](#实验步骤)
- [实验结果与分析](#实验结果与分析)
- [使用说明](#使用说明)
- [项目结构](#项目结构)
- [参考](#参考)

---

## 项目简介

本实验使用 **Pix2Pix**（条件生成对抗网络，cGAN）在 **Facades** 建筑立面数据集上进行图像到图像的翻译任务，分别尝试了 **BtoA**（真实建筑图 → 抽象标注图）和 **AtoB**（抽象标注图 → 真实建筑图）两种方向，并使用可视化脚本对生成结果进行对比分析。

Pix2Pix 是由 Phillip Isola 等人于 2017 年提出的通用图像翻译框架，能够学习输入图像到输出图像的映射，广泛应用于草图着色、语义分割转真实图像、图像修复等任务。

---

## 环境配置

| 依赖 | 版本要求 | 用途 |
|------|----------|------|
| Python | 3.8+ | 开发语言 |
| PyTorch | 1.12+ | 深度学习框架 |
| torchvision | 0.13+ | 图像处理工具 |
| numpy | 1.21+ | 数值计算 |
| Pillow | 9.0+ | 图像加载 |
| matplotlib | 3.5+ | 结果可视化 |
| visdom | — | 训练可视化（可选） |
| wandb | — | 实验跟踪（可选） |

> **注意：** 训练依赖 [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 官方仓库。可视化脚本仅需基础依赖即可运行。

详细依赖见 [`requirements.txt`](requirements.txt)。

---

## 实验目的

1. 理解图像到图像翻译的基本原理
2. 掌握 Pix2Pix 生成对抗网络的结构
3. 能够使用开源实现进行训练与图像翻译测试
4. 熟悉使用小型数据集在本地环境训练图像生成模型

---

## Pix2Pix 原理简介

Pix2Pix 是一种条件生成对抗网络，其核心结构包括：

- **生成器（Generator）：** 基于 U-Net 架构，将输入图像编码再解码为输出图像，通过跳跃连接（skip connections）保留低层细节信息。
- **判别器（Discriminator）：** 基于 PatchGAN 架构，将图像划分为若干 patches 并判断每个 patch 的真伪，鼓励生成器产生更精细的纹理。
- **损失函数：** 结合 $L_1$ 损失（鼓励生成图像与真实图像接近）和对抗损失（鼓励生成图像更逼真）。

---

## 数据集说明

- **数据集：** Facades（建筑立面数据集）
- **内容：** 包含建筑正立面的真实照片与对应的语义标注图
- **来源：** [pix2pix 官方数据集](https://cmp.felk.cvut.cz/~tylecr1/facade/)
- **数据划分：**
  - A 域：语义标注图（抽象的建筑结构标签图）
  - B 域：真实建筑照片
- **训练方向：**
  - **BtoA（对比实验）：** 真实建筑图 → 抽象标注图
  - **AtoB（对比实验）：** 抽象标注图 → 真实建筑图

---

## 实验步骤

### 1. 下载官方仓库与数据集

```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
cd pytorch-CycleGAN-and-pix2pix
```

数据集放置于 `./datasets/facades/` 目录。

### 2. 训练模型

#### BtoA 方向（真实建筑 → 抽象标注）

```bash
python train.py \
    --dataroot ./datasets/facades \
    --name facades_pix2pix \
    --model pix2pix \
    --direction BtoA \
    --n_epochs 20 \
    --n_epochs_decay 0 \
    --batch_size 1 \
    --gpu_ids -1 \
    display_id -1 \
    --save_epoch_freq 1
```

#### AtoB 方向（抽象标注 → 真实建筑）

将上述命令中的 `--direction BtoA` 改为 `--direction AtoB`，其余参数不变。

> **训练耗时：** 约 3 小时（CPU 环境）

### 3. 测试模型

```bash
python test.py \
    --dataroot ./datasets/facades \
    --name facades_pix2pix \
    --model pix2pix \
    --direction BtoA \
    --gpu_ids -1
```

测试结果保存在 `results/facades_pix2pix/test_latest/images/` 目录下。

### 4. 可视化结果

运行 [`实验17-pix2pix.py`](./实验17-pix2pix.py) 脚本，以三列并排方式展示生成结果：

```bash
python 实验17-pix2pix.py
```

默认展示第 1 组结果（n=1），可在脚本中修改变量 `n` 查看不同组。

---

## 实验结果与分析

### BtoA（真实建筑 → 抽象标注）

- 整体结构已能识别，但细节不够清晰
- 存在一定的空间扭曲感
- 从抽象到具体的重建难度较大

### AtoB（抽象标注 → 真实建筑）

- 建筑的基本结构能够较好地生成
- 颜色还原度较高，接近真实照片
- **效果优于 BtoA**，更接近 Ground Truth

### 对比分析

| 方向 | 效果 | 特点 |
|------|------|------|
| BtoA | 模糊，结构不清晰 | 真实→抽象，重建难度大 |
| AtoB | 清晰，还原度较高 | 抽象→真实，语义还原度好 |

### 改进方向

- 添加超分辨率模块以提升生成图像质量
- 增加训练轮次或使用更好的硬件（GPU）
- 尝试不同的网络结构变体

---

## 使用说明

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练前的准备

下载 [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 仓库，并将 Facades 数据集放入 `./datasets/facades/`。

### 3. 启动训练

参考上方实验步骤中的命令，根据需要选择 BtoA 或 AtoB 方向。

### 4. 可视化检测结果

训练完成后，直接运行：

```bash
python 实验17-pix2pix.py
```

确保 `result_dir` 路径指向正确的测试结果目录。

---

## 项目结构

```
├── 实验17-pix2pix.py               # 结果可视化脚本
├── README.md                       # 项目说明（本文件）
├── requirements.txt                # 依赖清单
├── pytorch-CycleGAN-and-pix2pix/   # 官方仓库（需自行克隆）
│   ├── train.py                    # 训练入口
│   ├── test.py                     # 测试入口
│   ├── models/                     # 模型定义
│   └── datasets/                   # 数据集目录
│       └── facades/                # Facades 数据集
└── results/                        # 测试结果保存目录
    └── facades_pix2pix/
        └── test_latest/
            └── images/             # 可视化脚本读取路径
```

---

## 参考

- [Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) (Isola et al., CVPR 2017)
- [pytorch-CycleGAN-and-pix2pix 官方仓库](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [Facades 数据集](https://cmp.felk.cvut.cz/~tylecr1/facade/)
