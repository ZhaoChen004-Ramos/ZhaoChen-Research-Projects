Vbat = 3.7  # 电池电压 (V)

P_static = 0.065

screen = {"L": 0.1, "M": 0.3, "H": 0.5}
cpu    = {"L": 0.05, "M": 0.5, "H": 1.5}

network = {
    "L": 0.25 + 0.04*0.01 + 0.04*0.05,
    "M": 0.25 + 0.04*0.1  + 0.04*0.5,
    "H": 0.25 + 0.04*1.0  + 0.04*5.0
}

gps = {"L": 0.0, "M": 0.12, "H": 0.12}

modes = ["L", "M", "H"]

for m in modes:
    P_total = P_static + screen[m] + cpu[m] + network[m] + gps[m]
    I = P_total / Vbat
    print(f"{m} 模式: 总功率 = {P_total:.3f} W, 电流 = {I:.3f} A")
