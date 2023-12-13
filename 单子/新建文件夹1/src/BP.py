import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 准备数据
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_excel(os.path.join(base_dir, 'data', '1原始数据.xlsx'))
df = pd.DataFrame(data)
df = df[df['组别'] == '对照组']
df = df.drop(columns=['组别', '县域', '年份'])
# print(df.head())
new_column_names = {'产业结构-第一产业占比\n%': '第一产业占比',
                    '教育人力资本\n（财政教育支出/财政支出）': '教育人力资本',
                    '种植结构\n（粮食作物播种面积/农作物播种面积）': '种植结构',
                    '农业技术水平\n（县域农机总动力）\n万千瓦时': '农业技术水平',
                    '地方政府财政收入\n（万元）': '地方政府财政收入',
                    '经济发展水平\n（人均GDP）': '经济发展水平',
                    '人均资本投入\n（农村人均用电量）': '人均资本投入',}
df = df.rename(columns=new_column_names)

# 目标变量和特征变量
target = '农村居民年人均可支配收入\n（元）'
features = df.columns.drop(target)

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)
X_train = df[features]
y_train = df[target]

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# 创建BP神经网络模型
bp_nn = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=0)

# 训练模型
bp_nn.fit(X_train_scaled, y_train)

# 获取每个特征的重要性
# 由于神经网络不直接提供特征重要性，我们可以通过观察权重来间接估计
# 获取输入层到第一个隐藏层的权重
weights = bp_nn.coefs_[0]

# 计算权重的平均绝对值，用作特征重要性的代理
importance = np.mean(np.abs(weights), axis=1)

# 创建特征重要性的DataFrame
nn_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})

# 对特征重要性进行排序
nn_importance_df = nn_importance_df.sort_values(by='Importance', ascending=False)

# 可视化特征重要性
sns.set_style('darkgrid')
font1 = {'family': [ 'Times New Roman', 'SimSun'], 'weight': 'normal', 'size': 14}
plt.rc('font', **font1)
plt.rcParams["axes.unicode_minus"] = False
plt.subplots_adjust(left=0.3)
sns.barplot(x='Importance', y='Feature', data=nn_importance_df)
plt.title('BP特征重要性', fontproperties=font1)
plt.xlabel('重要性', fontproperties=font1)
plt.gca().set_ylabel('')
# plt.ylabel('变量', fontproperties=font1)
plt.savefig(os.path.join(base_dir, 'res', 'BP.tif'))
plt.show()
