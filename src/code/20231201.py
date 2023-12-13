from matplotlib import pyplot as plt
import numpy as np

# 参数设置
plt.style.use('seaborn-darkgrid')
plt.rcParams['font.family'] = 'Times New Roman, SimSun'

# 数据
countries = ['挪威', '德国', '中国', '美国', '瑞典']
gold_medal = np.array([16, 12, 9, 8, 8])
silver_medal = np.array([8, 10, 4, 10, 5])
bronze_medal = np.array([13, 5, 2, 7, 5])
width = 0.3

#绘图
plt.bar(countries, gold_medal, color='gold', label='金牌',bottom=silver_medal + bronze_medal,width=width)
plt.bar(countries, silver_medal, color='silver', label='银牌', bottom=bronze_medal,width=width)
plt.bar(countries, bronze_medal, color='#A0522D', label='铜牌',width=width)

#设置y轴标签，图例和文本值
plt.ylabel('奖牌数')
plt.legend(loc='upper right')
for i in range(len(countries)):
    max_y = bronze_medal[i]+silver_medal[i]+gold_medal[i]
    plt.text(countries[i], max_y, max_y, va="bottom", ha="center")

plt.show()