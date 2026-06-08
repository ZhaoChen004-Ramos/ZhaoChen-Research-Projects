import numpy as np
import pandas as pd
dataset=pd.read_excel('附件1.xlsx')
dataset=dataset.values
dataset1=[]
dataset2=[]
for i in range(0,len(dataset),1):
    a=dataset[i][1]/100
    dataset1.append(a)
for i in range(0, len(dataset), 1):
    a = dataset[i][0]
    dataset2.append(a)
#导入牛顿迭代法数值求解器
from scipy.optimize import fsolve
def problem(n1,reflection_known,theta0):
    theta0=np.radians(theta0)
    sin_theta0=np.sin(theta0)
    cos_theta0=np.cos(theta0)

    #计算RS
    a=np.sqrt(n1**2-sin_theta0**2)-cos_theta0
    a=a**4
    b=(1-n1**2)**2
    rs=a/b

    #计算RP
    c=np.sqrt(n1**2-sin_theta0**2)
    d=cos_theta0*(n1**2)
    e=d/c+1
    f=1-2/e
    f=pow(f,2)
    rp=f

    r=(rs+rp)/2
    return r-reflection_known
#将加载的反射率加载到这个列表当中
reflection_known=dataset1
list_n=[]
for i in range(0,len(dataset1),1):
    #定义入射角，附件1=10，附件2=15
    theta0 = 10
    n1_guess = 2
    reflection_known_work=reflection_known[i]
    n1_solve = fsolve(problem, n1_guess, args=(reflection_known_work, theta0))
    list_n.append(n1_solve)
    print(f"求解得到的折射率 n1 = {n1_solve}")

x=dataset2
y=list_n
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 6))
plt.plot(x, y,color='orange',lw=1,marker='o',markersize=0.5,)
plt.xlabel('波数', fontsize=12, fontweight='bold')
plt.ylabel('折射率', fontsize=12, fontweight='bold')
plt.title('碳化硅外延层折射率与波数的关系', fontsize=14, fontweight='bold', pad=20)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
print(len(list_n))

