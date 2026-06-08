import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =========================
# 电池参数
# =========================
C_nominal = 4.5 * 3600
R0_ref = 0.05
Rc = 0.05
Cp = 2500

Ea = 30000
Rg = 8.314
T_ref = 298

# =========================
# 工况参数
# =========================
P = 0.467          # 低功耗
T = 273            # 常温 25°C

initial_SOCs = [1.0, 0.75, 0.5, 0.25]

def R0_T(T):
    return R0_ref * np.exp(Ea / Rg * (1/T - 1/T_ref))

R0 = R0_T(T)

# =========================
# OCV函数
# =========================
def OCV(soc):
    return (4.15117104*soc**5
           -12.89640731*soc**4
           +15.04794154*soc**3
           -7.36657513*soc**2
           +1.86009087*soc
           +3.36724086)

# =========================
# 更新后的电池ODE（电压闭环）
# =========================
def battery_ode(t, y):
    soc, Vp = y
    V_oc = OCV(soc)

    # 真实端电压 = Voc - Vp - I*R0
    # 恒功率：P = Vt * I → I = P / Vt
    # 联立得到 I
    Vt_guess = V_oc - Vp
    I = P / (Vt_guess + 1e-6)

    Vt = V_oc - Vp - I * R0
    I = P / (Vt + 1e-6)

    dSOC_dt = -I / C_nominal
    dVp_dt = -Vp / (Rc * Cp) + I / Cp
    return [dSOC_dt, dVp_dt]

# =========================
# SOC=0 事件
# =========================
def soc_zero_event(t, y):
    return y[0]

soc_zero_event.terminal = True
soc_zero_event.direction = -1

# =========================
# 🎨 画图
# =========================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"SOC vs Time (until SOC = 0)\n"
             f"Power={P}W  Temp={T}K",
             fontsize=16, fontweight='bold')

axes = axes.flatten()

for ax, soc0 in zip(axes, initial_SOCs):

    sol = solve_ivp(
        battery_ode,
        [0, 200000],
        [soc0, 0.0],
        method='LSODA',
        events=soc_zero_event,
        max_step=5
    )

    t_hours = sol.t / 3600
    soc = sol.y[0]

    ax.plot(t_hours, soc, linewidth=2)
    ax.set_title(f"Initial SOC = {int(soc0*100)}%", fontweight='bold')
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("SOC")
    ax.grid(True, alpha=0.3)

    # 打印SOC清零时间
    if sol.t_events[0].size > 0:
        t_end = sol.t_events[0][0] / 3600
        print(f"Initial SOC {int(soc0*100)}% → Time to SOC=0: {t_end:.2f} hours")
    else:
        print(f"Initial SOC {int(soc0*100)}% → SOC did not reach 0")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
