import numpy as np
import matplotlib.pyplot as plt

# 生成时间序列
t = np.linspace(0, 10, 100)

# 功耗模型函数
P_static = np.ones_like(t) * 0.1  # 静态功耗：常数
P_screen = 0.3 * t  # 屏幕功耗：假设与时间线性相关
P_CPU = 0.5 + (1.6 - 0.5) * np.power(t/10, 1)  # CPU功耗：假设随时间的某种非线性变化
P_network = 0.2 + 0.1 * t  # 网络功耗：假设与时间线性相关

# 创建子图（2x2）
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 静态功耗
axs[0, 0].plot(t, P_static, label="Power Consumption_static", color="blue")
axs[0, 0].set_title("Static Power Consumption")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Power (W)")
axs[0, 0].legend()

# 屏幕功耗
axs[0, 1].plot(t, P_screen, label="Power Consumption_screen", color="green")
axs[0, 1].set_title("Screen Power Consumption")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Power (W)")
axs[0, 1].legend()

# CPU功耗
axs[1, 0].plot(t, P_CPU, label="Power Consumption_CPU", color="red")
axs[1, 0].set_title("CPU Power Consumption")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Power (W)")
axs[1, 0].legend()

# 网络功耗
axs[1, 1].plot(t, P_network, label="Power Consumption_network", color="purple")
axs[1, 1].set_title("Network Power Consumption")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Power (W)")
axs[1, 1].legend()

# 添加总标题
fig.suptitle("Basic Trend Curve of the Power Consumption Model", fontsize=16)

# 调整子图之间的间距
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 显示图表
plt.show()