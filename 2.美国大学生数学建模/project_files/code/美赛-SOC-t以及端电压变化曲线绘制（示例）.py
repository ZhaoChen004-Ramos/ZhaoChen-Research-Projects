import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =========================
# Battery Parameters
# =========================
C_nominal = 4.5 * 3600
R0_ref = 0.05
Rc = 0.05
Cp = 2500
V_cut = 3.4

Ea = 30000
Rg = 8.314
T_ref = 298

# ===== 只选择一个工况 =====
P = 2.675              # Low Power
T = 273           # 25°C (常温)

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

def battery_ode(t, y):
    soc, Vp = y
    V_oc = OCV(soc)
    I = P / (V_oc - Vp + 1e-6)
    dSOC_dt = -I / C_nominal
    dVp_dt = -Vp / (Rc * Cp) + I / Cp
    return [dSOC_dt, dVp_dt]

def cutoff_event(t, y):
    soc, Vp = y
    V_oc = OCV(soc)
    I = P / (V_oc - Vp + 1e-6)
    Vt = V_oc - Vp - I * R0
    return Vt - V_cut

cutoff_event.terminal = True
cutoff_event.direction = -1

R0 = R0_T(T)

# =========================
# Solve ODE
# =========================
sol = solve_ivp(
    battery_ode,
    [0, 230000],
    [1.0, 0.0],   # 初始 SOC=1, Vp=0
    method='LSODA',
    events=cutoff_event,
    max_step=5
)

# =========================
# Plot
# =========================
t_hours = sol.t / 3600
soc = sol.y[0]

plt.figure(figsize=(6,4))
plt.plot(t_hours, soc, linewidth=2)
plt.xlabel("Time (hours)")
plt.ylabel("SOC")
plt.title("SOC vs Time (High Power, 0°C)")
plt.grid(True)
plt.tight_layout()
plt.show()










Vt = []
for soc, Vp in zip(sol.y[0], sol.y[1]):
    Voc = OCV(soc)
    I = P / (Voc - Vp + 1e-6)
    Vt.append(Voc - Vp - I*R0)

plt.figure()
plt.plot(sol.t/3600, Vt)
plt.axhline(3.4, color='r', linestyle='--')
plt.xlabel("Time (h)")
plt.ylabel("Terminal Voltage (V)")
plt.title("Terminal Voltage vs Time (High Power, 0°C)")
plt.grid(True)
plt.show()
