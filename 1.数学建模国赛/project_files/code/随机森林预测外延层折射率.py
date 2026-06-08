import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 解决中文显示问题
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#读取数据
df = pd.read_excel('附件1.xlsx')
wave_number = df['波数 (cm-1)'].values.reshape(-1, 1)
reflectivity = df['反射率 (%)'].values.reshape(-1, 1)
X = np.hstack([wave_number, reflectivity])

# 数据清洗：剔除反射率异常值（0-100%），确保特征有效性
valid_mask = (reflectivity >= 0) & (reflectivity <= 100)
valid_mask = valid_mask.flatten()
X = X[valid_mask]
wave_number_clean = wave_number[valid_mask]
reflectivity_clean = reflectivity[valid_mask]

#定义模拟折射率
sample_num = len(X)
#模拟折射率下限
n_base = 2
# 确保所有数组都是一维的
wave_number_flat = wave_number_clean.flatten()
n_wave_correction = (wave_number_flat - np.min(wave_number_flat)) / (
            np.max(wave_number_flat) - np.min(wave_number_flat))
n_noise = np.random.normal(0, 0.1, sample_num)
y_true = n_base + n_wave_correction * 3.5 + n_noise
y_true = np.clip(y_true, 2.0, 5.5)

#模型训练
wave_number_clean_flat = wave_number_clean.flatten()
X_train, X_test, y_train, y_test, wave_train, wave_test = train_test_split(
    X, y_true, wave_number_clean_flat, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#训练参数，可以修改
rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

#模型评估
y_train_pred = rf_model.predict(X_train_scaled)
y_test_pred = rf_model.predict(X_test_scaled)

# 计算评估指标（R²接近1、RMSE接近0为优）
print("\n" + "=" * 50)
print("折射率预测模型评估结果")
print("=" * 50)
print("训练集：")
print(f"  R²决定系数：{r2_score(y_train, y_train_pred):.4f}")
print(f"  RMSE均方根误差：{np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print("测试集：")
print(f"  R²决定系数：{r2_score(y_test, y_test_pred):.4f}")
print(f"  RMSE均方根误差：{np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print("=" * 50)

#可视化
# 按波数排序
sorted_idx = np.argsort(wave_test)
wave_test_sorted = wave_test[sorted_idx]  # x轴：1维
y_test_true_sorted = y_test[sorted_idx]   # y轴：1维
y_test_pred_sorted = y_test_pred[sorted_idx]  # 预测值：1维
# 绘图
plt.figure(figsize=(12, 6))
plt.scatter(wave_test_sorted, y_test_true_sorted,
            color='red', s=30, label='真实折射率（模拟）', alpha=0.7)
plt.plot(wave_test_sorted, y_test_pred_sorted,
         color='darkblue', linewidth=2.5, label='随机森林预测折射率', alpha=0.9)
plt.xlabel('波数 (cm⁻¹)', fontsize=12)
plt.ylabel('外延层折射率 n', fontsize=12)
plt.title('基于波数-反射率的折射率逆预测结果', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.ylim(2, 5.5)  # 匹配模拟折射率区间
plt.show()

#计算具体参数
def predict_refractive_index(wave_number_new, reflectivity_new, model, scaler):
    # 处理输入格式（转为2维特征）
    wave_new = np.array(wave_number_new).reshape(-1, 1)
    ref_new = np.array(reflectivity_new).reshape(-1, 1)
    X_new = np.hstack([wave_new, ref_new])
    # 标准化+预测
    X_new_scaled = scaler.transform(X_new)
    n_pred = model.predict(X_new_scaled)
    return n_pred

#根据波峰确定中获取的索引值，写入一个索引列表进行访问求解（基于之前的级数和索引列表），计算新的厚度
index_1=[245,491,1694,2078]
index_2=[249,498,1705,2109]
n1=[]
n2=[]
for i in range(0,4,1):
    wave_new = wave_number_clean[index_1[i]].flatten()
    ref_new = reflectivity_clean[index_1[i]].flatten()
    n_pred_new = predict_refractive_index(wave_new, ref_new, rf_model, scaler)
    n1.append(n_pred_new)
for i in range(0,4,1):
    wave_new = wave_number_clean[index_2[i]].flatten()
    ref_new = reflectivity_clean[index_2[i]].flatten()
    n_pred_new = predict_refractive_index(wave_new, ref_new, rf_model, scaler)
    n2.append(n_pred_new)
wave1=[19.3127,15.7135,8.2211,7.1351]
wave2=[19.2411,15.6306,8.1854,7.0599]
k1=[4.3659,5.3659,6.5703,7.5703]
k2=[4.3293,5.3293,6.2723,7.2723]
d1=[]
d2=[]
print(n1)
print(n2)
for i in range(0,4,1):
    #定义角度，附件1的话theta=10,附件2的话theta=15
    theta=10
    theta=np.radians(theta)
    theta=np.sin(theta)
    theta=pow(theta,2)
    n=n1[i]
    n=pow(n,2)
    #根据第一问的模型进行公式的输入，代入求解厚度d
    d=k1[i]*wave1[i]/(2*np.sqrt(n-theta))
    d1.append(d)
for i in range(0,4,1):
    #定义角度，附件1的话theta=10,附件2的话theta=15
    theta=15
    theta=np.radians(theta)
    theta=np.sin(theta)
    theta=pow(theta,2)
    n=n2[i]
    n=pow(n,2)
    #根据第一问的模型进行公式的输入，代入求解厚度d
    d=k2[i]*wave2[i]/(2*np.sqrt(n-theta))
    d2.append(d)
print(d1)
print(d2)