from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 准备数据
data = pd.read_csv(r'dataset.csv')
df = pd.DataFrame(data)

# 目标变量和特征变量
target = 'target'
features = df.columns.drop(target)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features].values, df[target].values, test_size=0.2, random_state=0)

# 重构训练集dataframe
train_X = pd.DataFrame(X_train, columns=features)
train_y = pd.DataFrame(y_train, columns=[target])
train = pd.concat([train_X, train_y],axis = 1)
print(train)

# 对训练集进行相关性分析
sns.set(font_scale=1.2)
plt.rc('font',family=['SimSun'], size=12)
plt.figure(figsize=(10, 8))
plt.subplots_adjust()
ax = sns.heatmap(train.corr(), annot=True, xticklabels=False, fmt=".2f")
ax.set_title('相关性热力图')  # 图标题
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()