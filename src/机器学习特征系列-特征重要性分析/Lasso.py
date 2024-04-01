import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso,LassoCV

# 准备数据
data = pd.read_csv(r'E:\数据杂坛\\UCI Heart Disease Dataset.csv')
df = pd.DataFrame(data)

# 目标变量和特征变量
target = 'target'
features = df.columns.drop(target)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)


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
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=lasso_coefficients_df)
plt.title('Feature Coefficients from Lasso Regression')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.show()
