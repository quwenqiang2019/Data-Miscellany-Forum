import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns



data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)
print(df.columns)
sns.set(font_scale=1.2)
plt.rc('font',family=['Times New Roman', 'SimSun'], size=12)
plt.scatter(df['age'], df['chol'])
plt.title('age与chol的关系')
plt.xlabel('age')
plt.ylabel('chol')
plt.tight_layout()
plt.show()


data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)
colors = ['red','blue']
target = df['target'].unique()

sns.set(font_scale=1.2)
plt.rc('font',family=['Times New Roman', 'SimSun'], size=12)
for i in range(len(target)):
    plt.scatter(df.loc[df.target == i, 'age'], df.loc[df.target==i,'chol'], s = 35, c = colors[i], label = i)
plt.title('age与chol的关系')
plt.xlabel('age')
plt.ylabel('chol')
plt.legend(loc='upper left')# 默认是左上方，
plt.tight_layout()
plt.show()


data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)
# 假设thalach的第三个特征展示为为气泡大小
fea = df['thalach']
plt.scatter(df['age'], df['chol'], s=fea/2, c='purple', alpha=0.4, edgecolors="grey",
            linewidth=2)
plt.xlabel('age')  # 横坐标轴标题
plt.ylabel('chol')  # 纵坐标轴标题
plt.title('s=thalach/2, c=purple', verticalalignment='bottom')
plt.tight_layout()
plt.show()
# 参数说明
# s：表征气泡大小的变量
# c：颜色，若想要彩色气泡，可以给c赋值，如c=fea
# alpha：不透明度
# edgecolors：气泡描边的颜色
# linewidth：气泡描边大小


data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)

sns.set(font_scale=1.2)
plt.rc('font',family=['Times New Roman', 'SimSun'], size=12)
ax = plt.subplot(projection = '3d')  # 创建一个三维的绘图工程
ax.scatter(df['age'], df['chol'], df['thalach'])
plt.tight_layout()
plt.show()


# 生成模拟数据
N=1000
x = np.random.normal(size=N)
y = x * 3 + np.random.normal(size=N)

# 计算样本点密度
xy = np.vstack([x,y])  #  将两个维度的数据叠加
z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度

# 按密度排序，将密度最大的点排在最后
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
sns.set(font_scale=1.2)
plt.rc('font',family=['Times New Roman', 'SimSun'], size=12)
fig, ax = plt.subplots()
plt.scatter(x, y,c=z, s=20,cmap='Spectral') # c表示标记的颜色
plt.colorbar()
plt.show()