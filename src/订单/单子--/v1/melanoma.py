import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 准备数据
data=pd.read_csv("melanoma.csv")
df=pd.DataFrame(data)

#设置目标变量和特征变量
target='N'
features=df.columns.drop(target)
print(features)

#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[[target]], test_size=0.2, random_state=0)

# 归一化
mm1 = MinMaxScaler()   # 特征进行归一化
X_train_m = mm1.fit_transform(X_train)
mm2 = MinMaxScaler()     # 标签进行归一化
y_train_m = mm2.fit_transform(y_train)


model = AdaBoostClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# 提取特征重要性
feature_importance = model.feature_importances_
feature_names = features

# 创建特征重要性的DataFrame
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

# 对特征重要性进行排序
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 可视化特征重要性
plt.figure(figsize=(11, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance(Coefficient)')
plt.xlabel('Weight Importance')
plt.ylabel('Variable')
plt.show()



