# 导入第三方包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
from scipy.stats import norm

base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(base_dir)

# 读取数据集
heart = pd.read_csv(os.path.join(base_dir, 'data', 'UCI Heart Disease Dataset.csv'))
# 检查年龄是否有缺失
any(heart.age.isnull())
# 不妨删除含有缺失年龄的观察
heart.dropna(subset=['age'], inplace=True)

# 设置图形的显示风格
plt.style.use('ggplot')
# 字体设置
config = {
    "font.family": 'Times New Roman, SimSun', # 衬线字体
    "font.size": 12, # 相当于小四大小
    "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False # 处理负号，即-号
}
plt.rcParams.update(config)
# 绘图：患者年龄的频数直方图
plt.hist(heart.age, # 绘图数据
        bins = 20, # 指定直方图的条形数为20个
        color = 'steelblue', # 指定填充色
        edgecolor = 'k', # 指定直方图的边界色
        label = '直方图' )# 为直方图呈现标签
plt.title('患者年龄的频数直方图')
plt.xlabel('年龄')
plt.ylabel('频数')
# 显示图例
plt.legend()
# 显示图形
plt.show()



# 绘图：年龄的累计频率直方图
plt.hist(heart.age, # 绘图数据
        bins = np.arange(heart.age.min(),heart.age.max(),5), # 指定直方图的组距
        density = True, # 设置为频率直方图
        cumulative = True, # 积累直方图
        color = 'steelblue', # 指定填充色
        edgecolor = 'k', # 指定直方图的边界色
        label = '直方图' )# 为直方图呈现标签

# 设置坐标轴标签和标题
plt.title('患者年龄的频率累计直方图')
plt.xlabel('年龄')
plt.ylabel('累计频率')
# 显示图例
plt.legend(loc = 'best')
# 显示图形
plt.show()



# 正态分布图
plt.hist(heart.age, # 绘图数据
        bins = np.arange(heart.age.min(),heart.age.max(),5), # 指定直方图的组距
        density = True, # 设置为频率直方图
        color = 'steelblue', # 指定填充色
        edgecolor = 'k') # 指定直方图的边界色

# 设置坐标轴标签和标题
plt.title('患者年龄频率直方图')
plt.xlabel('年龄')
plt.ylabel('频率')

# 生成正态曲线的数据
x1 = np.linspace(heart.age.min(), heart.age.max(), 1000)
normal = norm.pdf(x1, heart.age.mean(), heart.age.std())
# 绘制正态分布曲线
line1, = plt.plot(x1,normal,'r-', linewidth = 2)

# 生成核密度曲线的数据
kde = mlab.GaussianKDE(heart.age)
x2 = np.linspace(heart.age.min(), heart.age.max(), 1000)
# 绘制
line2, = plt.plot(x2,kde(x2),'g-', linewidth = 2)
# 显示图例
plt.legend([line1, line2],['正态分布曲线','核密度曲线'],loc='best')
# 显示图形
plt.show()




# 提取不同性别的年龄数据
age_female = heart.age[heart.sex == 0]
age_male = heart.age[heart.sex == 1]
# 设置直方图的组距
bins = np.arange(heart.age.min(), heart.age.max(), 2)
# 男性患者年龄直方图
plt.hist(age_male, bins = bins, label = '男性', color = 'steelblue', alpha = 0.7)
# 女性患者年龄直方图
plt.hist(age_female, bins = bins, label = '女性', alpha = 0.6)
# 设置坐标轴标签和标题
plt.title('患者年龄频数直方图')
plt.xlabel('年龄')
plt.ylabel('人数')
# 显示图例
plt.legend()
# 显示图形
plt.show()

