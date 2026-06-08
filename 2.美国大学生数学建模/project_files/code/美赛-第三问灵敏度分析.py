import numpy as np
from scipy.integrate import solve_ivp

# =========================
# 🔋 电池参数（基准）
# =========================
C_nominal_base = 4.5 * 3600
R0_ref_base = 0.05
Rc = 0.05
Cp = 2500
V_cut = 3.4

Ea = 30000
Rg = 8.314
T_ref = 298
T_use = 273  # 当前工况温度

# =========================
# ⚙️ 各子功耗（基准）
# =========================
params_base = {
    "CPU": 1.5,
    "Screen": 0.5,
    "Signal": 0.39,
    "Static": 0.065,
    "GPS": 0.12
}


# =========================
# 📈 电池模型函数
# =========================
def OCV(soc):
    # 拟合的开路电压曲线
    return (4.15117104 * soc ** 5
            - 12.89640731 * soc ** 4
            + 15.04794154 * soc ** 3
            - 7.36657513 * soc ** 2
            + 1.86009087 * soc
            + 3.36724086)


def R0_T(R0_ref):
    # Arrhenius 方程修正内阻
    return R0_ref * np.exp(Ea / Rg * (1 / T_use - 1 / T_ref))


# =========================
# 🔋 电池ODE系统
# =========================
def battery_ode(t, y, P_total, C_nominal, R0):
    soc, Vp = y
    Voc = OCV(soc)
    # 计算电流 I = P / V_terminal
    # 为避免代数环，这里简化计算电流
    I = P_total / (Voc - Vp + 1e-6)

    dSOC_dt = -I / C_nominal
    dVp_dt = -Vp / (Rc * Cp) + I / Cp
    return [dSOC_dt, dVp_dt]


def cutoff_event(t, y, P_total, C_nominal, R0):
    soc, Vp = y
    Voc = OCV(soc)
    I = P_total / (Voc - Vp + 1e-6)
    Vt = Voc - Vp - I * R0
    return Vt - V_cut


# 设置事件属性
cutoff_event.terminal = True
cutoff_event.direction = -1


# =========================
# ⏱ 计算关机时间函数
# =========================
def shutdown_time(C_nominal, R0_ref, powers):
    P_total_val = sum(powers.values())
    R0_val = R0_T(R0_ref)

    # 修正点：args 必须包含 ODE 和 Event 函数共同需要的所有额外参数
    sol = solve_ivp(
        battery_ode,
        [0, 200000],
        [1.0, 0.0],
        args=(P_total_val, C_nominal, R0_val),
        events=cutoff_event,
        max_step=10  # 适当放大步长提高灵敏度分析速度
    )

    if sol.t_events[0].size > 0:
        return sol.t_events[0][0] / 3600
    else:
        return np.nan


# =========================
# 🔹 1️⃣ 执行分析
# =========================
print("正在计算基准关机时间...")
T0 = shutdown_time(C_nominal_base, R0_ref_base, params_base)
print(f"✅ 基准关机时间 T0 = {T0:.3f} h\n")

print("正在进行灵敏度分析（请稍候）...")
sensitivity = {}

# 电池容量灵敏度
T_plus = shutdown_time(C_nominal_base * 1.1, R0_ref_base, params_base)
T_minus = shutdown_time(C_nominal_base * 0.9, R0_ref_base, params_base)
sensitivity["Capacity"] = (T_plus - T_minus) / (0.2 * T0)

# 内阻灵敏度
T_plus = shutdown_time(C_nominal_base, R0_ref_base * 1.1, params_base)
T_minus = shutdown_time(C_nominal_base, R0_ref_base * 0.9, params_base)
sensitivity["R0"] = (T_plus - T_minus) / (0.2 * T0)

# 各子模块功耗灵敏度
for key in params_base:
    powers_plus = params_base.copy()
    powers_minus = params_base.copy()
    powers_plus[key] *= 1.1
    powers_minus[key] *= 0.9

    T_plus = shutdown_time(C_nominal_base, R0_ref_base, powers_plus)
    T_minus = shutdown_time(C_nominal_base, R0_ref_base, powers_minus)

    sensitivity[key] = (T_plus - T_minus) / (0.2 * T0)

# =========================
# 📊 输出结果
# =========================
print("-" * 30)
print(f"{'参数名称':<10} | {'灵敏度系数':<10}")
print("-" * 30)
# 按灵敏度绝对值排序输出
for k, v in sorted(sensitivity.items(), key=lambda item: abs(item[1]), reverse=True):
    print(f"{k:<12} | {v:.3f}")