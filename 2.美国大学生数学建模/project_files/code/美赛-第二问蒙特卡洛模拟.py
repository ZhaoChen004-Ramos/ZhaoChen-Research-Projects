import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==============================
# 🎯 Monte Carlo Settings
# ==============================
N = 500
confidence = 0.95

# ==============================
# 🌡 固定温度（自行修改）
# ==============================
temperature = 313   # 273=0°C, 298=25°C, 313=40°C

Ea = 30000
Rg = 8.314
T_ref = 298

# ==============================
# 🔋 电池参数（含不确定性）
# ==============================
C_nominal_mean = 4.5 * 3600
C_nominal_std = 0.05 * C_nominal_mean

R0_mean = 0.05
R0_std = 0.05 * R0_mean

Rc = 0.05
Cp = 2500
V_cut = 3.4

# ==============================
# ⚙️ 子功耗基准值
# ==============================
P_static_base = 0.065
P_screen_base = 0.3
P_cpu_base = 0.78
P_signal_base = 0.3
P_GPS_base = 0.12

power_sigma1 = 0.05
power_sigma2 = 0.2
power_sigma3 = 0.2
power_sigma4 = 0.2
power_sigma5 = 0.05

# ==============================
# OCV 曲线
# ==============================
def OCV(soc):
    return (4.15117104*soc**5
           -12.89640731*soc**4
           +15.04794154*soc**3
           -7.36657513*soc**2
           +1.86009087*soc
           +3.36724086)

def R0_T(T, R0_sample):
    return R0_sample * np.exp(Ea/Rg * (1/T - 1/T_ref))

# ==============================
# 存储结果
# ==============================
shutdown_times = []
inputs_record = []

# ==============================
# 🔁 Monte Carlo 主循环
# ==============================
for _ in range(N):

    C_nominal = np.random.normal(C_nominal_mean, C_nominal_std)
    R0_sample = np.random.normal(R0_mean, R0_std)
    R0 = R0_T(temperature, R0_sample)

    P_cpu = np.random.normal(P_cpu_base, power_sigma3 * P_cpu_base)
    P_screen = np.random.normal(P_screen_base, power_sigma2 * P_screen_base)
    P_signal = np.random.normal(P_signal_base, power_sigma4 * P_signal_base)
    P_static = np.random.normal(P_static_base, power_sigma1 * P_static_base)
    P_GPS = np.random.normal(P_GPS_base, power_sigma5 * P_GPS_base)

    P_total = P_cpu + P_screen + P_signal + P_GPS + P_static
    params = (C_nominal, R0, P_total)

    # ===== 电池ODE =====
    def battery_ode(t, y, params):
        soc, Vp = y
        Cn, R0, P = params
        V_oc = OCV(soc)
        I = P / (V_oc - Vp + 1e-6)
        dSOC_dt = -I / Cn
        dVp_dt = -Vp / (Rc * Cp) + I / Cp
        return [dSOC_dt, dVp_dt]

    # ===== 截止事件：端电压=3.4V =====
    def voltage_cutoff_event(t, y, params):
        soc, Vp = y
        Cn, R0, P = params
        V_oc = OCV(soc)
        I = P / (V_oc - Vp + 1e-6)
        Vt = V_oc - Vp - I * R0
        return Vt - V_cut

    voltage_cutoff_event.terminal = True
    voltage_cutoff_event.direction = -1

    sol = solve_ivp(
        battery_ode,
        [0, 200000],
        [1.0, 0.0],
        args=(params,),
        events=voltage_cutoff_event,
        max_step=5
    )

    if sol.t_events[0].size > 0:
        t_end = sol.t_events[0][0] / 3600
    else:
        t_end = np.nan

    shutdown_times.append(t_end)
    inputs_record.append([C_nominal, R0_sample, P_cpu, P_screen, P_signal, P_GPS, P_static])

# ==============================
# 📊 统计分析
# ==============================
shutdown_times = np.array(shutdown_times)
valid = ~np.isnan(shutdown_times)
shutdown_times = shutdown_times[valid]
inputs_record = np.array(inputs_record)[valid]

mean_time = np.mean(shutdown_times)
std_time = np.std(shutdown_times)

z = 1.96
ci_low = mean_time - z * std_time
ci_high = mean_time + z * std_time

print(f"\n平均关机时间: {mean_time:.2f} 小时")
print(f"标准差: {std_time:.2f} 小时")
print(f"{int(confidence*100)}%置信区间: [{ci_low:.2f}, {ci_high:.2f}] 小时")

# ==============================
# 📈 直方图（条纹填充）
# ==============================
plt.figure(figsize=(7,5))
plt.hist(shutdown_times, bins=25, edgecolor='black', hatch='//')
plt.axvline(mean_time, linestyle='--', linewidth=2)
plt.title("Monte Carlo Battery Life Distribution 40℃", fontweight='bold')
plt.xlabel("Shutdown Time (hours)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==============================
# 🔍 相关性分析（7个变量）
# ==============================
labels = ["Capacity", "R0_ref", "CPU Power", "Screen Power", "Signal Power", "GPS Power", "Static Power"]

print("\n各输入变量与关机时间的相关系数：")
for i in range(len(labels)):
    corr = np.corrcoef(inputs_record[:, i], shutdown_times)[0,1]
    print(f"{labels[i]:<15}: {corr:.3f}")
