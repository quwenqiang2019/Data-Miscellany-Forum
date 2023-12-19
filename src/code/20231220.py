import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)
values = np.random.rand(10, 12)  # 自定义数据

x_ticks = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']
y_ticks = ['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10']  # 自定义横纵轴
ax = sns.heatmap(values, xticklabels=x_ticks, yticklabels=y_ticks, annot=True,fmt ='0.2g')
ax.set_title('Heatmap')  # 图标题
ax.set_xlabel('x label')  # x轴标题
ax.set_ylabel('y label')
plt.show()
# figure = ax.get_figure()
# figure.savefig('sns_heatmap.jpg')  # 保存图片



import pandas as pd
# 准备数据
data = pd.read_csv(r'E:\数据杂坛\\UCI Heart Disease Dataset.csv')
df = pd.DataFrame(data)

sns.set(font_scale=1.2)
plt.rc('font',family=['Times New Roman', 'SimSun'], size=12)
plt.subplots_adjust()
ax = sns.heatmap(df.corr(), annot=True, fmt=".2f")
ax.set_title('相关性热力图')  # 图标题
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# figure = ax.get_figure()
# figure.savefig('sns_heatmap.jpg', bbox_inches='tight')