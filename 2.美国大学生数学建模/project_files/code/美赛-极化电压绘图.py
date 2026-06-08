import numpy as np
import matplotlib.pyplot as plt

# ========== 参数 ==========
Rc = 0.05      # 极化电阻 (Ohm)
Cp = 2500      # 极化电容 (F)
tau = Rc * Cp  # 时间常数 (s)

# 三种模式下的电流 (A)
currents = {
    "Low Power Mode": 0.126,
    "Medium Power Mode": 0.34,
    "High Power Mode": 0.732
}

# 时间范围（秒）
t = np.linspace(0, 600, 1000)  # 模拟 10 分钟

# ========== 计算并绘图 ==========
plt.figure(figsize=(8,5))

for mode, I in currents.items():
    Vp = I * Rc * (1 - np.exp(-t / tau))  # 极化电压解析解
    plt.plot(t, Vp, label=mode)

plt.xlabel("Time (s)")
plt.ylabel("Polarization Voltage Vp (V)")
plt.title("Polarization Voltage under Different Power Modes")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
