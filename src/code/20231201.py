from matplotlib import pyplot as plt
import numpy as np

# 参数设置
plt.style.use('seaborn-darkgrid')
plt.rcParams['font.family'] = 'Times New Roman, SimSun'

# 数据
classes = ['一班', '二班', '三班', '四班', '五班']
language = np.array([87, 85, 89, 81, 78])
math = np.array([85, 98, 84, 79, 82])
english = np.array([83, 85, 82, 87, 78])
width = 0.3

#绘图
plt.bar(classes, language, color='gold', label='语文',bottom=math + english,width=width)
plt.bar(classes, math, color='silver', label='数学', bottom=english,width=width)
plt.bar(classes, english, color='#A0522D', label='英语',width=width)

#设置y轴标签，图例和文本值
plt.ylabel('平均分')
plt.legend(loc='upper right')
for i in range(len(classes)):
    max_y = english[i]+math[i]+language[i]
    plt.text(classes[i], max_y, max_y, va="bottom", ha="center")

plt.show()