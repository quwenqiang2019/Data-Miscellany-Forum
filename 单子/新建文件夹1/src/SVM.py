import os
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
使用线性支持向量机（SVM）模型和递归特征消除（RFE）进行特征重要性分析
'''


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

# 训练线性支持向量机（SVR）模型
svr = SVR(kernel='linear')
svr.fit(X_train, y_train)

# 使用递归特征消除（RFE）进行特征选择
selector = RFE(svr, n_features_to_select=1, step=1)
selector = selector.fit(X_train, y_train)

# 获取特征排名
feature_ranking = selector.ranking_

# 创建特征排名的DataFrame
ranking_df = pd.DataFrame({'Feature': features, 'Ranking': feature_ranking})

# 对特征排名进行排序
ranking_df = ranking_df.sort_values(by='Ranking')

# 可视化特征排名
sns.set_style('darkgrid')
font1 = {'family': [ 'Times New Roman', 'SimSun'], 'weight': 'normal', 'size': 14}
plt.rc('font', **font1)
plt.rcParams["axes.unicode_minus"] = False
plt.subplots_adjust(left=0.3)
sns.barplot(x='Ranking', y='Feature', data=ranking_df)
plt.title('SVM特征重要性', fontproperties=font1)
plt.xlabel('重要性', fontproperties=font1)
plt.gca().set_ylabel('')
# plt.ylabel('变量', fontproperties=font1)
plt.savefig(os.path.join(base_dir, 'res', 'SVM.tif'))
plt.show()
