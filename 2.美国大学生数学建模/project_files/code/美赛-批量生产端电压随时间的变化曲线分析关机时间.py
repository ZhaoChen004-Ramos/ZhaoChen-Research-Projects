import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =========================
# 🔋 电池参数
# =========================
C_nominal = 4.5 * 3600
R0_ref = 0.05
Rc = 0.05
Cp = 2500
V_cut = 3.4

Ea = 30000
Rg = 8.314
T_ref = 298

# =========================
# ⚙️ 可调参数区域
# =========================
initial_soc = 1           # 初始电量
temperature = 313           # 低温 0°C

power_modes = {
    "Low Power": 0.467,
    "Medium Power": 1.259,
    "High Power": 2.675
}

# =========================
# OCV 曲线
# =========================
def OCV(soc):
    return (4.15117104*soc**5
           -12.89640731*soc**4
           +15.04794154*soc**3
           -7.36657513*soc**2
           +1.86009087*soc
           +3.36724086)

def R0_T(T):
    return R0_ref * np.exp(Ea / Rg * (1 / T - 1 / T_ref))

R0 = R0_T(temperature)

# =========================
# 电池微分方程
# =========================
def battery_ode(t, y, P):
    soc, Vp = y
    Voc = OCV(soc)
    Veff = Voc - Vp

    # 解二次方程求电流（恒功率闭环）
    disc = Veff**2 - 4 * R0 * P
    if disc <= 0:
        I = Veff / (2 * R0)  # 数值保护
    else:
        I = (Veff - np.sqrt(disc)) / (2 * R0)

    dSOC_dt = -I / C_nominal
    dVp_dt = -Vp / (Rc * Cp) + I / Cp
    return [dSOC_dt, dVp_dt]


# 终止事件：端电压达到截止电压
def cutoff_event(t, y, P):
    soc, Vp = y
    Voc = OCV(soc)
    Veff = Voc - Vp

    disc = Veff**2 - 4 * R0 * P
    if disc <= 0:
        I = Veff / (2 * R0)
    else:
        I = (Veff - np.sqrt(disc)) / (2 * R0)

    Vt = Voc - Vp - I * R0
    return Vt - V_cut


cutoff_event.terminal = True
cutoff_event.direction = -1

# =========================
# 🎨 画图
# =========================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f"Terminal Voltage vs Time (Initial SOC=100%, {temperature-273}°C)",
             fontsize=16, fontweight='bold')

for ax, (mode, P) in zip(axes, power_modes.items()):

    sol = solve_ivp(
        battery_ode,
        [0, 200000],
        [initial_soc, 0.0],
        args=(P,),
        method='LSODA',
        events=cutoff_event,
        max_step=5
    )

    t_hours = sol.t / 3600
    Vt_list = []

    for soc, Vp in zip(sol.y[0], sol.y[1]):
        Voc = OCV(soc)
        Veff = Voc - Vp
        disc = Veff ** 2 - 4 * R0 * P
        if disc <= 0:
            I = Veff / (2 * R0)
        else:
            I = (Veff - np.sqrt(disc)) / (2 * R0)

        Vt_list.append(Voc - Vp - I * R0)

    ax.plot(t_hours, Vt_list, linewidth=2)
    ax.axhline(V_cut, color='r', linestyle='--', alpha=0.6)
    ax.set_title(mode, fontweight='bold')
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Terminal Voltage (V)")
    ax.grid(True, alpha=0.3)

    # 打印关机时间
    if sol.t_events[0].size > 0:
        t_shutdown = sol.t_events[0][0] / 3600
        print(f"{mode} | Shutdown Time (Vt=3.4V): {t_shutdown:.2f} hours")
    else:
        print(f"{mode} | Voltage did not reach cutoff")

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
