import numpy as np
import pandas as pd
#根据需要可以修改附件1和附件2
dataset=pd.read_excel('附件4.xlsx')
dataset=dataset.values
dataset1=[]
dataset2=[]
#写入反射率和波数数据
for i in range(0,len(dataset),1):
    a=dataset[i][0]
    dataset1.append(a)
for i in range(0,len(dataset),1):
    a=dataset[i][1]
    dataset2.append(a)
#通过可视化图可以确认波数区域的最高峰位置，找到四个峰，然后手动定义搜索区间（索引），写入列表的范围中，并获取索引
feng_1=dataset2[210:1247]
feng_1_max=max(feng_1)
feng_1_max_index=feng_1.index(feng_1_max)
feng_1_max_index+=210

feng_2=dataset2[1247:1900]
feng_2_max=max(feng_2)
feng_2_max_index=feng_2.index(feng_2_max)
feng_2_max_index+=1247

feng_3=dataset2[2082:2492]
feng_3_max=max(feng_3)
feng_3_max_index=feng_3.index(feng_3_max)
feng_3_max_index+=2082

feng_4=dataset2[2700:3320]
feng_4_max=max(feng_4)
feng_4_max_index=feng_4.index(feng_4_max)
feng_4_max_index+=2700

print(feng_1_max_index,feng_2_max_index,feng_3_max_index,feng_4_max_index)
#转化波数为波长，单位为微米，进制为10000
wave1=1/dataset1[feng_1_max_index]*10000
wave2=1/dataset1[feng_2_max_index]*10000
wave3=1/dataset1[feng_3_max_index]*10000
wave4=1/dataset1[feng_4_max_index]*10000

#定义波长列表，已知的反射率列表（变为小数），定义波数列表
list_wave=[wave1,wave2,wave3,wave4]
list_reflection=[dataset2[feng_1_max_index]/100,dataset2[feng_2_max_index]/100,
                 dataset2[feng_3_max_index]/100,dataset2[feng_4_max_index]/100]
list_v=[dataset1[feng_1_max_index],dataset1[feng_2_max_index],
                 dataset1[feng_3_max_index],dataset1[feng_4_max_index]]
print(list_wave)
print(list_reflection)
print(list_v)

#计算四个级数，相邻的两个峰，第二个峰的级数认为是第一个峰级数+1
k1=list_wave[1]/(list_wave[0]-list_wave[1])
k2=k1+1
k3=list_wave[3]/(list_wave[2]-list_wave[3])
k4=k3+1
list_k=[k1,k2,k3,k4]

#通过菲涅尔公式折射率求解.py求解出来的反射率，写入列表，n11_list代表附件1的，n12_list代表附件2的
n11_list=[3.78649691,3.98764999,2.15392013,2.29839155]
n12_list=[4.09477971,4.33531017,2.21749227,2.36275089]

#迭代循环，计算四个厚度
for i in range(0,4,1):
    #定义角度，附件1的话theta=10,附件2的话theta=15
    theta=15
    theta=np.radians(theta)
    theta=np.sin(theta)
    theta=pow(theta,2)
    n=n12_list[i]
    n=pow(n,2)
    #根据第一问的模型进行公式的输入，代入求解厚度d
    d=list_k[i]*list_wave[i]/(2*np.sqrt(n-theta))
    print(d)