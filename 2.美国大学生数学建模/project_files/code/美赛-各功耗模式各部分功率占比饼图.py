import matplotlib.pyplot as plt
import numpy as np

# =========================
# 中功耗模式下的功耗数据（根据之前的文献给定）
# =========================
total_power = 2.675  # 中功耗模式总功耗

# 各子功耗（单位：W）
sub_power = {
    "Static Power": 0.065,   # 静态功耗
    "Screen Power": 0.5,     # 屏幕功耗
    "CPU Power": 1.5,       # CPU功耗
    "Network Power": 0.39,   # 网络功耗
    "GPS Power": 0.12         # GPS功耗
}

# 计算每个子功耗占比
power_values = {k: v / total_power for k, v in sub_power.items()}

# =========================
# 绘制 2D 饼图
# =========================
labels = sub_power.keys()
sizes = power_values.values()
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'orange']

# 饼图设置
fig, ax = plt.subplots(figsize=(8, 6))

# 饼图绘制
ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90, wedgeprops={'edgecolor': 'black'})

# 让饼图更圆
ax.axis('equal')

# 设置标题
ax.set_title("Power Distribution in High Power Mode (2.675W)", fontsize=16, fontweight='bold')

# 显示图表
plt.tight_layout()
plt.show()
