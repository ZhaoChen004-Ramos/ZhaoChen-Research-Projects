import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r"deg5.xlsx"

# 读取正确工作表
df = pd.read_excel(file_path, sheet_name='Channel_1_1')

# 只保留需要的列
df = df[['Test_Time(s)', 'Voltage(V)', 'Current(A)', 'Step_Index']]

# 只保留真正放电阶段
df = df[df['Step_Index'] ==5]

# 转为 numpy
time = df['Test_Time(s)'].values
voltage = df['Voltage(V)'].values
current = np.abs(df['Current(A)'].values)  # 放电取绝对值

print("有效放电数据点数：", len(time))

# ================= SOC 计算 =================
C_nominal_Ah = 4.2
C_nominal = C_nominal_Ah * 3600

dt = np.diff(time, prepend=time[0])
soc = 1 - np.cumsum(current * dt) / C_nominal
soc = np.clip(soc, 0, 1)

time_hours = time / 3600

# ================= SOC-t =================
plt.figure(figsize=(6,4))
plt.plot(time_hours, soc, linewidth=2)
plt.xlabel("Time (hours)")
plt.ylabel("SOC")
plt.title("Experimental SOC vs Time (5°C, Constant Current Discharge)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ================= V-t =================
plt.figure(figsize=(6,4))
plt.plot(time_hours, voltage, linewidth=2)
plt.xlabel("Time (hours)")
plt.ylabel("Terminal Voltage (V)")
plt.title("Experimental Voltage vs Time (5°C, Constant Current Discharge)")
plt.grid(True)
plt.tight_layout()
plt.show()
