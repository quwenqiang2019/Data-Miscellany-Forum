import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
import seaborn as sns
# 准备数据
data = pd.read_csv(r'dataset.csv')
df = pd.DataFrame(data)
print(df)
# 目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# 进行 ANOVA 分析，通过计算每个特征与目标变量之间的f统计量，来判断哪些特征与目标变量最相关。
f_scores, p_values = f_classif(X_train, y_train)

# 打印每个特征的 F 值和 p 值
for i, feature_name in enumerate(features):
    print(f"特征 '{feature_name}' 的 F 值：{f_scores[i]}, p 值：{p_values[i]}")

# 可视化特征重要性
sns.set(font_scale=1.2)
plt.rc('font',family=['Times New Roman', 'SimSun'], size=12)
plt.figure(figsize=(10, 6))
plt.bar(features, f_scores, color='skyblue')
plt.title("特征重要性：ANOVA F 值")
plt.xlabel("特征")
plt.ylabel("F 值")
plt.xticks(rotation=45)
plt.show()

