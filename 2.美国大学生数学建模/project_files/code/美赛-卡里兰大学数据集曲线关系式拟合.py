import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
file_path = r"02_24_2016_SP20-1_0C_lowcurrentOCV.xls"
df = pd.read_excel(file_path,sheet_name='Channel_1-005_1')

print(df.columns.tolist())

# 提取需要的列
voltage = df["Voltage(V)"].values
capacity = df["Discharge_Capacity(Ah)"].values

# 去掉NaN
mask = ~np.isnan(voltage) & ~np.isnan(capacity)
voltage = voltage[mask]
capacity = capacity[mask]

# 总容量取最大值
Q_total = np.max(capacity)

# 计算 SOC
soc = 1 - capacity / Q_total

# 排序（SOC从小到大）
idx = np.argsort(soc)
soc = soc[idx]
voltage = voltage[idx]

plt.plot(soc, voltage)
plt.xlabel("SOC")
plt.ylabel("OCV (V)")
plt.title("Raw OCV-SOC Data")
plt.grid(True)
plt.show()




# 5阶多项式拟合
coeffs = np.polyfit(soc, voltage, 5)

# 生成拟合函数
ocv_poly = np.poly1d(coeffs)

# 画图对比
soc_fit = np.linspace(0, 1, 200)
voltage_fit = ocv_poly(soc_fit)

plt.plot(soc, voltage, '.', label="Raw Data")
plt.plot(soc_fit, voltage_fit, '-', label="Fitted Curve")
plt.xlabel("SOC")
plt.ylabel("OCV (V)")
plt.title("Fitted OCV-SOC Curve")
plt.legend()
plt.grid(True)
plt.show()

print("拟合多项式系数：")
print(coeffs)
