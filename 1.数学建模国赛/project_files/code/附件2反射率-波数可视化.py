import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
dataset=pd.read_excel('附件2.xlsx')
dataset=dataset.values
dataset1=[]
dataset2=[]
#分别写入反射率和波数
for i in range(0,len(dataset),1):
    a=dataset[i][0]
    dataset1.append(a)
for i in range(0,len(dataset),1):
    a=dataset[i][1]
    dataset2.append(a)
x = dataset1
y = dataset2
plt.figure(figsize=(12, 6))
plt.plot(x, y,color='orange',lw=1,marker='o',markersize=0.5,)
plt.xlabel('波数', fontsize=12, fontweight='bold')
plt.ylabel('反射率', fontsize=12, fontweight='bold')
plt.title('碳化硅外延层波数与反射率关系图', fontsize=14, fontweight='bold', pad=20)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
