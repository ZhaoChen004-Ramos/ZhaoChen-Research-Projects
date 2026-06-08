import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# 读取数据（请根据实际文件路径修改）
file_path = 'C:/Users/HP/Desktop/Demo/数据/02_24_2016_SP20-1_0C_lowcurrentOCV.xls'  # 修改为您的文件路径
# 根据您的文件，数据在 'Channel_1-005_1' sheet
df = pd.read_excel(file_path, sheet_name='Channel_1-005_1', header=0)

# 提取 Step 6 的放电数据（主要放电过程）
step6_data = df[df['Step_Index'] == 6].copy()

# 转换为数值类型
voltage = pd.to_numeric(step6_data['Voltage(V)'], errors='coerce')
discharge_capacity = pd.to_numeric(step6_data['Discharge_Capacity(Ah)'], errors='coerce')

# 去除无效值
valid_mask = (~np.isnan(voltage)) & (~np.isnan(discharge_capacity))
voltage = voltage[valid_mask]
discharge_capacity = discharge_capacity[valid_mask]

# 计算 SOC = 1 - Q/Qmax
Q_max = discharge_capacity.max()
soc = 1 - (discharge_capacity / Q_max)

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 使用紫色线条，与参考图一致
ax.plot(soc, voltage, color='purple', linewidth=1.5, label='OCV Curve')

# 设置坐标轴标签（与参考图一致）
ax.set_xlabel('State of Charge (SOC)', fontsize=11)
ax.set_ylabel('Voltage(V)', fontsize=11)
ax.set_title('Voltage vs State of Charge (SOC)', fontsize=12, pad=15)

# 设置坐标轴范围（根据参考图风格调整）
ax.set_xlim(0, 1)
ax.set_ylim(3.4, 4.2)

# 设置刻度（主刻度和次刻度）
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2])

# 添加网格（参考图没有显示网格，如需添加可取消注释）
ax.grid(True, which='both', linestyle='--', alpha=0.3)

# 美化边框：只保留左和下边框，或者保持简洁
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 输出关键参数供参考
print(f"Max Capacity (Qmax): {Q_max:.6f} Ah")
print(f"Voltage range: {voltage.min():.3f} V - {voltage.max():.3f} V")
print(f"Data points: {len(voltage)}")