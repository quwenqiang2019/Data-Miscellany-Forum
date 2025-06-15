import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

# 数据
data = pd.DataFrame(pd.read_excel('工作簿1.xlsx'))
print(data.head())

# 设置画布
sns.set(font_scale=1.2)
plt.rc('font',family=['Times New Roman', 'SimSun'], size=12)

# 绘制工业废水的折线
ax1 = sns.lineplot(data=data, x='年份', y='工业废水（万吨）', color='blue', marker='o', label='工业废水（万吨）')

# 创建第二个Y轴：耕田面积的折线
ax2 = plt.gca().twinx()
sns.lineplot(data=data, x='年份', y='耕田面积（千公顷）', color='green', marker='*', ax=ax2, label='耕田面积（千公顷）')

# 设置Y轴标签
ax1.set_ylabel('工业废水（万吨）', color='blue')
ax2.set_ylabel('耕田面积（千公顷）', color='green')

# 添加图例
plt.title('2000-2021年工业废水与耕田面积变化')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 显示图形
plt.savefig('biquadratic_chart.jpg', bbox_inches='tight', dpi=600)
plt.show()