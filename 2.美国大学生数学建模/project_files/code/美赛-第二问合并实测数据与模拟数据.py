import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =====================================================
# ⚙️ 统一工况参数（这里改，全局生效）
# =====================================================
temperature_C = 40
temperature_K = 273.15 + temperature_C
power_mode_name = "High Power"
P = 2.675                     # 功率 W（和你模型一致）
C_nominal_model = 4.5 * 3600  # 模型容量
C_nominal_exp = 4.2 * 3600    # 实测电池容量
V_cut = 3.4

# =====================================================
# 🔬 一、读取实测数据
# =====================================================
file_path = r"deg40.xlsx"
df = pd.read_excel(file_path, sheet_name='Channel_1_1')
df = df[['Test_Time(s)', 'Voltage(V)', 'Current(A)', 'Step_Index']]
df = df[df['Step_Index'] == 5]

time_exp = df['Test_Time(s)'].values
voltage_exp = df['Voltage(V)'].values
current_exp = np.abs(df['Current(A)'].values)

dt = np.diff(time_exp, prepend=time_exp[0])
soc_exp = 1 - np.cumsum(current_exp * dt) / C_nominal_exp
soc_exp = np.clip(soc_exp, 0, 1)
time_exp_h = time_exp / 3600

# =====================================================
# 🧠 二、模型仿真
# =====================================================
R0_ref = 0.05
Rc = 0.05
Cp = 2500
Ea = 30000
Rg = 8.314
T_ref = 298

def OCV(soc):
    return (4.15117104*soc**5
           -12.89640731*soc**4
           +15.04794154*soc**3
           -7.36657513*soc**2
           +1.86009087*soc
           +3.36724086)

def R0_T(T):
    return R0_ref * np.exp(Ea / Rg * (1 / T - 1 / T_ref))

R0 = R0_T(temperature_K)

def battery_ode(t, y):
    soc, Vp = y
    Voc = OCV(soc)
    I = P / (Voc - Vp + 1e-6)
    dSOC_dt = -I / C_nominal_model
    dVp_dt = -Vp / (Rc * Cp) + I / Cp
    return [dSOC_dt, dVp_dt]

def cutoff_event(t, y):
    soc, Vp = y
    Voc = OCV(soc)
    I = P / (Voc - Vp + 1e-6)
    Vt = Voc - Vp - I * R0
    return Vt - V_cut

cutoff_event.terminal = True
cutoff_event.direction = -1

sol = solve_ivp(battery_ode, [0, 20000], [1.0, 0.0],
                method='LSODA', events=cutoff_event, max_step=5)

time_model_h = sol.t / 3600
soc_model = sol.y[0]

Vt_model = []
for soc, Vp in zip(sol.y[0], sol.y[1]):
    Voc = OCV(soc)
    I = P / (Voc - Vp + 1e-6)
    Vt_model.append(Voc - Vp - I * R0)

# =====================================================
# 🎨 三、SOC-t 对比图
# =====================================================
plt.figure(figsize=(7,5))
plt.plot(time_exp_h, soc_exp, linewidth=2.5, label="Experimental SOC")
plt.plot(time_model_h, soc_model, linewidth=2.5, linestyle='--', label="Model SOC")

plt.xlabel("Time (hours)", fontsize=12, fontweight='bold')
plt.ylabel("SOC", fontsize=12, fontweight='bold')
plt.title(f"SOC vs Time Comparison ({temperature_C}°C, {power_mode_name})",
          fontsize=14, fontweight='bold')

plt.grid(alpha=0.3)
plt.legend(frameon=True)
plt.tight_layout()
plt.show()

# =====================================================
# 🎨 四、V-t 对比图
# =====================================================
plt.figure(figsize=(7,5))
plt.plot(time_exp_h, voltage_exp, linewidth=2.5, label="Experimental Voltage")
plt.plot(time_model_h, Vt_model, linewidth=2.5, linestyle='--', label="Model Voltage")

plt.axhline(V_cut, color='r', linestyle=':', label="Cutoff Voltage")

plt.xlabel("Time (hours)", fontsize=12, fontweight='bold')
plt.ylabel("Terminal Voltage (V)", fontsize=12, fontweight='bold')
plt.title(f"Voltage vs Time Comparison ({temperature_C}°C, {power_mode_name})",
          fontsize=14, fontweight='bold')

plt.grid(alpha=0.3)
plt.legend(frameon=True)
plt.tight_layout()
plt.show()
