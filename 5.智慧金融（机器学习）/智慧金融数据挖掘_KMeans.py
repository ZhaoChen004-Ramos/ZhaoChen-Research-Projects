#导入必要的模块
import numpy as np#导入numpy库进行各种数学计算
import pandas as pd#导入pandas库进行各种数据分析
import matplotlib.pyplot as plt#导入matplotlib库进行数据的可视化操作
from sklearn.preprocessing import StandardScaler#导入StandardScaler库用于数据标准化，缩放至均值为0，方差为1
from sklearn.cluster import KMeans#导入Kmeans算法，用于将数据进行聚类
from sklearn.metrics import silhouette_score #导入silhouette_score进行轮廓系数的计算，从而评估聚类类数

#读取智慧金融的数据集
dataset = pd.read_csv("C:/Users/HP/Desktop/credit_card.csv")#利用pandas库读取csv文件

#查看数据集情况并进行一定的统计分析
dataset_description_df = dataset.describe()#假设dataset是一个DataFrame，describe用来生成描述性统计信息，包括均值、标准
#差、最小值、四分位数、中位数和最大值
print(dataset_description_df)#打印对应的信息

#查看相关性-热力图
import seaborn as sns#导入seaborn库，基于matplotlib的数据可视化库
numerical_dataset=dataset.drop(columns=['CUST_ID'])#由于ID无用，所以从dataset中删除了这一列
corrmat=numerical_dataset.corr()#计算所有数值型列的相关性矩阵
plt.figure(figsize=(15, 15))#设置图形的大小为15*15英寸
plt.xticks(rotation=90)#将x轴刻度标签旋转90度
plt.yticks(rotation=90)#将y轴刻度标签旋转90度
sns.heatmap(corrmat, vmax=1.0, square=True)#使用heatmap绘制相关性矩阵的热力图，vmax设置颜色条的最大值，True使每个相关系数
#占据一个正方形
plt.show()#显示图形

#查看相关性-皮尔森系数
corr_data = numerical_dataset.corr(method='pearson')#从DataFrame中计算相关系数，1为完全正相关，corr完成计算
print(corr_data)#打印皮尔森相关系数

#检查缺失数据
def get_missing_data_summary():
    dataset_na = dataset.isnull().sum()#计算数据集中每一列的缺失值数量
    dataset_na = dataset_na.drop(dataset_na[dataset_na == 0].index).sort_values(ascending=False)#移除没有缺失值的列
    #并按缺失值数量降序排序
    missing_data = pd.DataFrame({'Missing Count' :dataset_na})#将结果转化为一个DataFrame对象
    return missing_data#定义函数的返回值
missing_data = get_missing_data_summary()#获取数据的缺失数据
print(missing_data)#MINIMUM_PAYMENTS有313个缺失数据，CREDIT_LIMIT有1个缺失数据

#处理缺失数据
dataset=dataset.drop(dataset[dataset['CREDIT_LIMIT'].isnull()].index)#生成布尔序列，并获取缺失值的索引，丢弃缺失值并
#返回新的DataFrame
dataset= dataset.drop(dataset[dataset['MINIMUM_PAYMENTS'].isnull()].index)#生成布尔序列，并获取另外一列的缺失值，并返回
#了新的对象

#移除无用的列
X=dataset.iloc[:,1:].values#使用iloc选择器从第二列开始选择所有，相当于移除第一列，并且用.values将DataFrame转化为Numpy数组

#使用PCA降维
from sklearn.decomposition import PCA#导入PCA主成分分析的方法
pca = PCA(n_components = None)#创建新的PCA对象，并保留所有原始维度，计算每个主成分的方差比率
X_pca = pca.fit_transform(X)#拟合PCA模型，对numpy数组进行PCA变换，获得降维之后的数据X_pca
explained_variance_ratio = pca.explained_variance_ratio_#获取每一个主成分的方差比率

#画出累计方差解释图，用于选择新生成的自变量的个数
plt.figure()#创建一个新的图形
plt.plot(np.cumsum(pca.explained_variance_ratio_), c='orange')#绘制解释图，并将每个成分的方差比率求和展示
plt.xlabel('number of components')#设置x轴的标签为压缩后的数据维度
plt.ylabel('cumulative explained variance')#设置y轴的标签为累计方差
plt.show()#展示图形

# 选择新生成的自变量的个数
pca = PCA(n_components = 2)#指定保留两个主成分
X_pca = pca.fit_transform(X)#对数据X进行PCA主成分分析

# 特征缩放
sc_X = StandardScaler()#初始一个标准化器对象，使其均值为0方差为1
X_scaled = sc_X.fit_transform(X_pca)#对主成分分析之后的数据进行特征

# 训练K-Means模型并打印轮廓系数，确定K-Means中K的个数
for i in range(2, 21):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init=10, max_iter=300, random_state = 0)#初始化聚类对象
    #分组数等于i，算法将进行10次并选取最优结果，最大迭代次数为300
    kmeans.fit(X_scaled)#对标准化后的数据进行聚类分析
    y_kmeans = kmeans.predict(X_scaled)#预测数据点的聚类标签
    silhouette = silhouette_score(X_scaled, y_kmeans)#计算对应的轮廓系数
    print('当聚类个数是%d时，对应的轮廓系数是%.4f' %(i, silhouette))#打印对应的轮廓系数

#使用K=4建立K-Means模型
kmeans = KMeans(n_clusters = 4, init = 'k-means++', n_init=10, max_iter=300, random_state = 0)#初始化聚类对象
kmeans.fit(X_scaled)#对标准化后的数据进行聚类分析
y_kmeans = kmeans.predict(X_scaled)#获取预测点的标签
indices_of_0=[index for index, value in enumerate(y_kmeans) if value == 0]
print(indices_of_0)
indices_of_1=[index for index, value in enumerate(y_kmeans) if value == 1]
print(indices_of_1[0])
indices_of_2=[index for index, value in enumerate(y_kmeans) if value == 2]
print(indices_of_2[0])
indices_of_3=[index for index, value in enumerate(y_kmeans) if value == 3]
print(indices_of_3[0])

#可视化聚类效果
plt.figure()#创建新的图形
plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 0')
#绘制散点图，绘制第1个聚类中两个主成分，以红色标记
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 1')
#绘制散点图，绘制第2个聚类中两个主成分，以蓝色标记
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 2')
#绘制散点图，绘制第3个聚类中两个主成分，以绿色标记
plt.scatter(X_scaled[y_kmeans == 3, 0], X_scaled[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 3')
#绘制散点图，绘制第4个聚类中两个主成分，以蓝绿色标记
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
#绘制每一个聚类的中心
plt.title('Clusters of customers')#设置图表题目
plt.xlabel('pca1')#设置x轴标签为pca1
plt.ylabel('pca2')#设置y轴标签为pca2
plt.legend()#显示图例
plt.show()#显示图表