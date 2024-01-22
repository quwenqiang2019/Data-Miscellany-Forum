from matplotlib import pyplot as plt
import numpy as np

# 参数设置
plt.style.use('seaborn-darkgrid')
plt.rcParams['font.family'] = 'Times New Roman, SimSun'

# 数据
classes = ['一班', '二班', '三班', '四班', '五班']
language = [87, 85, 89, 81, 78]
math = [85, 98, 84, 79, 82]
english = [83, 85, 82, 87, 78]

# 将横坐标班级先替换为数值
x = np.arange(len(classes))
width = 0.2
language_x = x
math_x = x + width
english_x = x + 2 * width
# 绘图
plt.bar(language_x, language, width=width, color='gold', label='语文')
plt.bar(math_x,math,width=width,color="silver",label="数学")
plt.bar(english_x,english,width=width, color="saddlebrown",label="英语")
#将横坐标数值转换为班级
plt.xticks(x + width, labels=classes)

#显示柱状图的高度文本
for i in range(len(classes)):
    plt.text(language_x[i],language[i], language[i],va="bottom",ha="center",fontsize=8)
    plt.text(math_x[i],math[i], math[i],va="bottom",ha="center",fontsize=8)
    plt.text(english_x[i],english[i], english[i],va="bottom",ha="center",fontsize=8)

#显示图例
plt.legend(loc="upper right")
plt.show()
