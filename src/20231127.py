import pandas as pd

# 读取文本信息
df = pd.DataFrame(pd.read_csv('E:\数据杂坛\datasets\\1127-test-data.txt', header=None))
print(df.head(6))

import re

# 运用正则表达式提取「性别」信息
sex = df.applymap(lambda x: re.search(r'男|女', x).group())

# 年龄
age = df.applymap(lambda x: re.findall(r'\d+', x)[0]).transform(pd.to_numeric)

# 合并性别和年龄数据
df_concat = pd.concat([sex, age], axis=1)
df_concat.columns = ['性别', '年龄']

# 年龄分段
bins = [0, 50, 60, 70, 80, 100]
labels = ['<50', '50-59', '60-69', '70-79', '>=80']
df_concat['年龄段'] = pd.cut(df_concat.年龄, bins=bins, labels=labels, right=False)

print(df_concat)

# 导入绘图库
import matplotlib.pyplot as plt

# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['simhei']

# 按性别分组计数
sex_count = df_concat.groupby('性别')['性别'].count()

# 绘制饼图
fig1 = plt.figure(figsize=(8, 8))
ax1 = fig1.add_subplot(1, 1, 1)
patches, l_text, p_text = plt.pie(sex_count, labels=sex_count.index, autopct='%1.1f%%')

# 设置字体大小
for t in l_text:
    t.set_size(20)
for t in p_text:
    t.set_size(20)

# 图表标题
plt.title('新型肺炎死亡病例的性别占比')

plt.show()


# 按年龄段分组计数
age_count = df_concat.groupby('年龄段')['年龄'].count()

# 绘制条形图
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(1, 1, 1)

# 设置字体大小
plt.rcParams.update({'font.size': 20})
ax2.set_xlabel('年龄', fontsize=20)
ax2.set_ylabel('人数', fontsize=20)

plt.bar(age_count.index, age_count)

# 图表标题
plt.title('不同年龄段的死亡人数')

plt.show()