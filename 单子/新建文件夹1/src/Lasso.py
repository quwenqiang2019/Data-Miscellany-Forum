import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso,LassoCV

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


Lambdas = np.logspace(-5, 2, 200)    #10的-5到10的2次方
# 构造空列表，用于存储模型的偏回归系数
lasso_cofficients  = []
for Lambda in Lambdas:
    lasso = Lasso(alpha  = Lambda, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_cofficients.append(lasso.coef_)
# 绘制Lambda与回归系数的关系
plt.plot(Lambdas, lasso_cofficients)
# 对x轴作对数变换
plt.xscale('log')
# 设置折线图x轴和y轴标签
plt.xlabel('Lambda')
plt.ylabel('Cofficients')
# 显示图形
plt.show()


# 训练Lasso模型（带有交叉验证）
lasso = LassoCV(cv=5)
lasso.fit(X_train, y_train)

# 输出最佳的lambda值
lasso_best_alpha = lasso.alpha_

# 采用最佳的Lasso模型
lasso_best = Lasso(alpha  = lasso_best_alpha,  max_iter=10000)
lasso_best.fit(X_train, y_train)

# 获取特征的系数
lasso_coefficients = lasso_best.coef_

# 创建特征系数的DataFrame
lasso_coefficients_df = pd.DataFrame({'Feature': features, 'Coefficient': lasso_coefficients})

# 对特征系数进行排序
lasso_coefficients_df = lasso_coefficients_df.sort_values(by='Coefficient', ascending=False)

# 可视化特征系数
sns.set_style('darkgrid')
font1 = {'family': [ 'Times New Roman', 'SimSun'], 'weight': 'normal', 'size': 14}
plt.rc('font', **font1)
plt.rcParams["axes.unicode_minus"] = False
plt.subplots_adjust(left=0.3)
sns.barplot(x='Coefficient', y='Feature', data=lasso_coefficients_df)
plt.title('Lasso特征重要性', fontproperties=font1)
plt.xlabel('重要性', fontproperties=font1)
plt.gca().set_ylabel('')
# plt.ylabel('变量', fontproperties=font1)
plt.savefig(os.path.join(base_dir, 'res', 'Lasso.tif'))
plt.show()
