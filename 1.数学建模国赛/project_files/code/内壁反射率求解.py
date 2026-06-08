import pandas as pd
import numpy as np
import math

# 读取文件
dataset = pd.read_excel('附件3.xlsx')
dataset = dataset.values
dataset1 = []
dataset2 = []

# 写入反射率和波数数据
for i in range(0, len(dataset), 1):
    a = (1 / dataset[i][0]) * 10000
    dataset1.append(a)

for i in range(0, len(dataset), 1):
    a = dataset[i][1] / 100
    dataset2.append(a)

mlambdam = [26.2505, 26.0334]
sin_lambda_2 = []

for i in range(0, len(dataset1), 1):
    xiangweicha_work = 2 * (np.pi) * mlambdam[0] / dataset1[i]
    xiangweicha_work /= 2
    xiangweicha_work = np.radians(xiangweicha_work)
    xiangweicha_work = np.sin(xiangweicha_work)
    xiangweicha_work = pow(xiangweicha_work, 2)
    sin_lambda_2.append(xiangweicha_work)


def solve_rho(R, S):
    if not (0 <= R <= 1):
        raise ValueError("R的取值范围必须在[0, 1]之间")
    if not (0 <= S <= 1):
        raise ValueError("S(sin²(δ/2))的取值范围必须在[0, 1]之间")
    if R == 0:
        raise ValueError("R不能为0，否则方程无解")

    a = R
    b = - (2 * R + 4 * S * (1 - R))
    c = R

    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        return []

    sqrt_d = math.sqrt(discriminant)
    rho1 = (-b + sqrt_d) / (2 * a)
    rho2 = (-b - sqrt_d) / (2 * a)

    valid_rhos = []
    for rho in [rho1, rho2]:
        if 0 <= rho <= 1:
            valid_rhos.append(round(rho, 8))

    return list(set(valid_rhos))


if __name__ == "__main__":
    R_list = dataset2
    S_list = sin_lambda_2
    results = []
    for i in range(len(R_list)):
        R = R_list[i]
        S = S_list[i]
        try:
            rhos = solve_rho(R, S)
            results.append({
                "索引": i,
                "R": R,
                "S": S,
                "ρ值": rhos if rhos else "无有效解"
            })
        except ValueError as e:
            results.append({
                "索引": i,
                "R": R,
                "S": S,
                "ρ值": f"计算错误: {str(e)}"
            })

    list_p = []
    for i in range(0, len(dataset1), 1):
        df = results[i]
        values = df['ρ值']
        list_p.append(values)


    print(list_p)
    # 过滤掉字符串元素，只保留可以比较的数值类型元素
    valid_list_p = []
    for element in list_p:
        if isinstance(element, (int, float)):
            valid_list_p.append(element)
        elif isinstance(element, list):
            valid_list_p.extend([x for x in element if isinstance(x, (int, float))])

    small_indices = []
    small_values = []
    for index, element in enumerate(list_p):
        if isinstance(element, (int, float)) and element < 0.05:
            small_indices.append(index)
            small_values.append(element)
        elif isinstance(element, list):
            for sub_index, sub_element in enumerate(element):
                if isinstance(sub_element, (int, float)) and sub_element < 0.05:
                    small_indices.append(index)
                    small_values.append(sub_element)

    if small_indices:
        print(f"小于 0.05 的值的索引为 {small_indices}，对应的值为 {small_values}")
    else:
        print("列表中没有小于 0.05 的值")