import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ========== 参数设置 ==========
R0_REF = 0.05  # 基础内阻 (Ω)
EA = 30000  # 活化能 (J/mol)
R_GAS = 8.314  # 理想气体常数 (J/(mol·K))
T_REF_C = 25  # 参考温度 (°C)
T_REF_K = T_REF_C + 273.15  # 参考温度 (K)


def R0_temperature(T_celsius):
    """阿伦尼乌斯定律计算内阻"""
    T_kelvin = T_celsius + 273.15
    exponent = (EA / R_GAS) * (1 / T_kelvin - 1 / T_REF_K)
    return R0_REF * np.exp(exponent)


# ========== 图1: 基础趋势图（单曲线）==========
def plot_basic_trend():
    temps = np.linspace(-20, 60, 500)
    resistances = R0_temperature(temps)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 主曲线
    ax.plot(temps, resistances, 'b-', linewidth=3, label=f'$E_a = {EA}$ J/mol')

    # 参考点
    ax.scatter([25], [0.05], color='red', s=150, zorder=5, edgecolors='black', linewidth=2)
    ax.annotate('Reference Point\(25°C, 0.05Ω)', xy=(25, 0.05), xytext=(35, 0.08),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # 关键温度点
    critical_temps = [0, -10, 40]
    for t in critical_temps:
        r_val = R0_temperature(t)
        ax.scatter([t], [r_val], color='black', s=80, zorder=5)
        ax.text(t, r_val + 0.015, f'{t}°C\{r_val:.3f}Ω', ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 区域填充
    ax.axvspan(-20, 15, alpha=0.15, color='blue', label='Cold Zone')
    ax.axvspan(15, 35, alpha=0.15, color='green', label='Optimal Zone')
    ax.axvspan(35, 60, alpha=0.1, color='red', label='Warm Zone')

    ax.set_xlabel(r'Temperature $T$ (°C)', fontsize=12)
    ax.set_ylabel(r'Internal Resistance $R_0$ ($\Omega$)', fontsize=12)
    ax.set_title(
        r'Arrhenius Law: $R_0 = R_{0,ref} \cdot \exp\left[\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right]$',
        fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-20, 60)

    return fig, ax


# ========== 图2: 多参数对比（不同活化能）==========
def plot_Ea_comparison():
    temps = np.linspace(-20, 60, 300)

    fig, ax = plt.subplots(figsize=(10, 6))

    Ea_values = [20000, 30000, 40000]
    colors = ['green', 'blue', 'red']

    for ea, color in zip(Ea_values, colors):
        # 临时修改EA计算
        T_k = temps + 273.15
        exponent = (ea / R_GAS) * (1 / T_k - 1 / T_REF_K)
        r_vals = R0_REF * np.exp(exponent)
        ax.plot(temps, r_vals, color=color, linewidth=2.5,
                label=f'$E_a = {ea}$ J/mol')

    ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5)
    ax.scatter([25], [0.05], color='black', s=100, zorder=5)

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel(r'$R_0$ ($\Omega$)')
    ax.set_title('Sensitivity Analysis: Effect of Activation Energy $E_a$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


# ========== 图3: 相对变化率（百分比）==========
def plot_relative_change():
    temps = np.linspace(-20, 60, 300)
    r_vals = R0_temperature(temps)
    change_pct = ((r_vals / 0.05) - 1) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(temps, change_pct, 'darkgreen', linewidth=3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 填充正负区域
    ax.fill_between(temps, change_pct, 0, where=(temps < 25),
                    color='blue', alpha=0.3, label='Capacity Loss (Cold)')
    ax.fill_between(temps, change_pct, 0, where=(temps > 25),
                    color='orange', alpha=0.3, label='Improved Efficiency (Warm)')

    # 标记关键百分比
    key_temps = [-10, 0, 40]
    for t in key_temps:
        idx = np.argmin(np.abs(temps - t))
        y_val = change_pct[idx]
        ax.scatter([t], [y_val], color='red', s=60, zorder=5)
        ax.text(t, y_val + 5, f'{y_val:.0f}%', ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Temperature (°C)')
    ax.ylabel('Resistance Change Relative to 25°C (%)')
    ax.set_title('Impact of Temperature on Battery Internal Resistance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


# ========== 图4: 双轴对比（内阻+有效容量）==========
def plot_resistance_and_capacity():
    temps = np.linspace(-20, 45, 300)  # 限制温度范围避免过热区
    r_vals = R0_temperature(temps)

    # 假设有效容量与内阻成反比（简化模型）
    # 实际应用中可能是更复杂的非线性关系
    capacity_factor = 1 / (1 + 0.5 * (r_vals / 0.05 - 1))  # 归一化到25°C=1.0

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Internal Resistance (Ω)', color=color1)
    ln1 = ax1.plot(temps, r_vals, color=color1, linewidth=3, label='Resistance')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.scatter([25], [0.05], color='red', s=80, zorder=5)

    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Relative Capacity Factor', color=color2)
    ln2 = ax2.plot(temps, capacity_factor, color=color2, linewidth=3,
                   linestyle='--', label='Capacity Factor')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.3, 1.1)

    # 合并图例
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right')

    ax1.set_title('Temperature Effect: Resistance vs Effective Capacity')
    ax1.grid(True, alpha=0.3)

    return fig, (ax1, ax2)


# ========== 运行绘图 ==========
if __name__ == "__main__":
    # 选择要绘制的图（取消注释需要的）

    # 图1: 基础趋势（最常用，推荐）
    fig1, ax1 = plot_basic_trend()
    plt.savefig('fig1_basic_trend.png', dpi=200, bbox_inches='tight')
    plt.show()

    # 图2: 参数敏感性分析
    # fig2, ax2 = plot_Ea_comparison()
    # plt.savefig('fig2_Ea_comparison.png', dpi=200, bbox_inches='tight')
    # plt.show()

    # 图3: 相对变化百分比
    # fig3, ax3 = plot_relative_change()
    # plt.savefig('fig3_relative_change.png', dpi=200, bbox_inches='tight')
    # plt.show()

    # 图4: 双轴对比
    # fig4, ax4 = plot_resistance_and_capacity()
    # plt.savefig('fig4_dual_axis.png', dpi=200, bbox_inches='tight')
    # plt.show()