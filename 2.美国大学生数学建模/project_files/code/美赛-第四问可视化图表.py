import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ================= 电池参数 =================
C_nominal = 4.5 * 3600
R0_ref = 0.05
Rc = 0.05
Cp = 2500
V_cut = 3.4

Ea = 30000
Rg = 8.314
T_ref = 298

# ================= 基准功耗 =================
P_cpu = 1.5
P_screen = 0.5
P_signal = 0.39
P_static = 0.065
P_GPS = 0.12

# ================= 电池模型 =================
def OCV(soc):
    return (4.15117104*soc**5
           -12.89640731*soc**4
           +15.04794154*soc**3
           -7.36657513*soc**2
           +1.86009087*soc
           +3.36724086)

def R0_T(T):
    return R0_ref * np.exp(Ea / Rg * (1 / T - 1 / T_ref))

def battery_ode(t, y, P):
    soc, Vp = y
    Voc = OCV(soc)
    I = P / (Voc - Vp + 1e-6)
    return [-I / C_nominal, -Vp/(Rc*Cp) + I/Cp]

def cutoff_event(t, y, P):
    soc, Vp = y
    Voc = OCV(soc)
    I = P / (Voc - Vp + 1e-6)
    Vt = Voc - Vp - I * R0_T(298)
    return Vt - V_cut

cutoff_event.terminal = True
cutoff_event.direction = -1

def shutdown_time(P_total):
    sol = solve_ivp(battery_ode, [0, 200000], [1.0, 0.0],
                    args=(P_total,), events=cutoff_event, max_step=5)
    return sol.t_events[0][0] / 3600

# ================= 五种情况 =================
P_base = P_cpu + P_screen + P_signal + P_static + P_GPS
P_screen_low = P_cpu + P_screen*0.9 + P_signal + P_static + P_GPS
P_cpu_low = P_cpu*0.9 + P_screen + P_signal + P_static + P_GPS
P_signal_low = P_cpu + P_screen + P_signal*0.9 + P_static + P_GPS
P_GPS_off = P_cpu + P_screen + P_signal + P_static + 0

# ================= 计算时间 =================
T_base = shutdown_time(P_base)
T_screen = shutdown_time(P_screen_low)
T_cpu = shutdown_time(P_cpu_low)
T_signal = shutdown_time(P_signal_low)
T_gps = shutdown_time(P_GPS_off)

times = [T_base, T_screen, T_cpu, T_signal, T_gps]

# 计算时间提升
improvements = [0,
                T_screen - T_base,
                T_cpu - T_base,
                T_signal - T_base,
                T_gps - T_base]

labels = ["Baseline",
          "Lower Screen",
          "Lower CPU",
          "Lower Network",
          "GPS Off"]

# ================= 画图 =================
plt.figure(figsize=(9,6))
bars = plt.bar(labels, improvements)
plt.axhline(0)
plt.ylabel("Increase in Usage Time (hours)")
plt.title("Battery Life Improvement from Single Optimization Actions", fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)

for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y,
             f"{y:.2f}h", ha='center', va='bottom')

plt.tight_layout()
plt.show()

# ================= 打印结果 =================
print(f"Baseline Time: {T_base:.2f} h")
print(f"↓ Screen: +{T_screen - T_base:.2f} h")
print(f"↓ CPU: +{T_cpu - T_base:.2f} h")
print(f"↓ Network: +{T_signal - T_base:.2f} h")
print(f"GPS Off: +{T_gps - T_base:.2f} h")
