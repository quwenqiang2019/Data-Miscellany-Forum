import matplotlib.pyplot as plt

# 准备数据
x = [1, 2, 3, 4, 5]  # x轴数据
y = [2, 6, 1, 3, 10]  # y轴数据

# 设置字体
plt.rcParams['font.family']='Times New Roman, SimSun'
# 绘制折线图
# plt.plot(x, y)
plt.plot(x, y,color='red', linestyle='--', marker='*')
# 添加标题和坐标轴标签
plt.title('折线图示例')
plt.xlabel('X轴')
plt.ylabel('Y轴')

# 显示图形
plt.show()



# encoding=utf-8
import matplotlib.pyplot as plt

# 月份
x1 = ['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08',
      '2017-09', '2017-10', '2017-11', '2017-12']

# 体重
y1 = [86, 85, 84, 80, 75, 70, 70, 74, 78, 70, 74, 80]

# 设置画布大小
plt.figure(figsize=(10, 7))
# 设置字体
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
plt.rc('font', **font1)
# 标题
plt.title("my weight", fontproperties=font1)

# 数据
plt.plot(x1, y1, label='weight changes', linewidth=3, color='r', marker='o',
         markerfacecolor='blue', markersize=14)

# 横坐标描述
plt.xlabel('month', fontproperties=font1)

# 纵坐标描述
plt.ylabel('weight', fontproperties=font1)

# 设置数字标签
for a, b in zip(x1, y1):
    plt.text(a, b+0.5, b, ha='center', va='bottom', fontproperties=font1)

plt.legend()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

x = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
highest = [12, 15, 18, 14, 16, 14, 10]
lowest = [6, 4, 8, 12, 10, 9, 7]

plt.plot(x, highest, "rs--", label="最高气温")
plt.plot(x, lowest, "rd--", label="最低气温")
for a, b in zip(x, highest):
    plt.text(a, b+1, b, ha='center', va='bottom')
    # 数据显示的横坐标、显示的位置高度、显示的数据值的大小
for a, b in zip(x, lowest):
    plt.text(a, b-2, b, ha='center', va='bottom')

# 绘图风格设置,使用seaborn库的API来设置样式
sns.set_style('darkgrid')
# # 设置字体
font1 = {'family': ['Times New Roman', 'SimSun'], 'weight': 'normal', 'size': 14}
plt.rc('font', **font1)
plt.rcParams["axes.unicode_minus"] = False

# x轴刻度标签设置
plt.xticks(x, fontproperties=font1)
# y轴标签数值范围设置
plt.ylim(0, 25)
# 标题设置
plt.title("一周气温变化趋势AAAA", fontproperties=font1)
plt.xlabel("星期", fontproperties=font1)
plt.ylabel("气温", fontproperties=font1)
# 图例设置
plt.legend()
plt.show()