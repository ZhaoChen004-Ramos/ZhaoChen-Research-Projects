import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================
# 🔋 基础电池参数
# ============================
C_nominal = 4.5 * 3600
R0_ref = 0.05
Rc, Cp = 0.05, 2500
V_cut = 3.4

Ea, Rg, T_ref = 30000, 8.314, 298

# ============================
# 🔥 环境与修正参数
# ============================
T_amb = 298  # 10°C (低温环境)
P_total = 8  # 8W (较高功耗)
k_peukert = 1.05
alpha_temp = 0.01


# ============================
# 🛠 物理模型函数
# ============================
def OCV(soc):
    soc = np.clip(soc, 0, 1)
    return (4.15117104 * soc ** 5 - 12.89640731 * soc ** 4 + 15.04794154 * soc ** 3
            - 7.36657513 * soc ** 2 + 1.86009087 * soc + 3.36724086)


def R0_T(T):
    return R0_ref * np.exp(Ea / Rg * (1 / T - 1 / T_ref))


def get_C_eff(I, T, use_correction):
    if not use_correction: return C_nominal
    I_ref = 4.5
    # Peukert修正：大电流使有效容量减少
    rate_factor = (max(I, 0.1) / I_ref) ** (k_peukert - 1)
    # 温度修正：低温使可用容量减少
    temp_factor = np.exp(alpha_temp * (T - T_ref))
    return C_nominal / rate_factor * temp_factor


# ============================
# 🔋 动力学方程
# ============================
def battery_dynamics(t, y, use_correction):
    soc, Vp = y
    Voc = OCV(soc)
    R_now = R0_T(T_amb if use_correction else T_ref)

    # 迭代计算电流 I = P / (Voc - Vp - I*R) 的简化形式
    I = P_total / (Voc - Vp + 1e-3)

    C_eff = get_C_eff(I, T_amb, use_correction)
    dSOC_dt = -I / C_eff
    dVp_dt = -Vp / (Rc * Cp) + I / Cp
    return [dSOC_dt, dVp_dt]


def cutoff_event(t, y, use_correction):
    soc, Vp = y
    Voc = OCV(soc)
    R_now = R0_T(T_amb if use_correction else T_ref)
    I = P_total / (Voc - Vp + 1e-3)
    Vt = Voc - Vp - I * R_now
    return Vt - V_cut


cutoff_event.terminal = True
cutoff_event.direction = -1

# ============================
# 🧮 运行对比仿真
# ============================
plt.figure(figsize=(12, 5))
ax1 = plt.subplot(1, 2, 1)  # SOC子图
ax2 = plt.subplot(1, 2, 2)  # Vt子图

configs = [
    {"label": "Corrected (25°C, Peukert+R_temp)", "correct": True, "color": "#2980b9", "ls": "-"}
]

for conf in configs:
    sol = solve_ivp(
        lambda t, y: battery_dynamics(t, y, conf["correct"]),
        [0, 50000], [1.0, 0.0],
        events=lambda t, y: cutoff_event(t, y, conf["correct"]),
        max_step=10, method='RK45'
    )

    t_h = sol.t / 3600
    soc_vals = sol.y[0]

    # 计算端电压 Vt
    vt_vals = []
    R_now = R0_T(T_amb if conf["correct"] else T_ref)
    for s, vp in zip(sol.y[0], sol.y[1]):
        voc = OCV(s)
        i = P_total / (voc - vp + 1e-3)
        vt_vals.append(voc - vp - i * R_now)

    # 绘图
    ax1.plot(t_h, soc_vals, label=conf["label"], color=conf["color"], ls=conf["ls"], lw=2)
    ax2.plot(t_h, vt_vals, label=conf["label"], color=conf["color"], ls=conf["ls"], lw=2)

    print(f"{conf['label']} Shutdown Time: {t_h[-1]:.2f} hours")

# ============================
# 📊 格式化图表
# ============================
ax1.set_title("SOC vs Time", fontweight='bold')
ax1.set_xlabel("Time (hours)")
ax1.set_ylabel("SOC")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.set_title("Terminal Voltage (Vt) vs Time", fontweight='bold')
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("Voltage (V)")
ax2.axhline(V_cut, color='red', ls=':', label="Cutoff Threshold")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()