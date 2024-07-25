import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
# Chla
data = pd.DataFrame({"冬季": [12.17243477, 11.36021739, 18.35030291, 17.15931569, 18.94613472, 16.43750166, 15.37389999, 10.84856549],
                     "春季": [16.36929306,22.83614637,20.4675646,28.12451086,28.131717,26.51940728,19.91029632,17.48587054],
                     "夏季": [20.0074298,16.45617457,19.00976795,30.69691155,29.66009154,26.608055,34.3336493,31.93616728],
                     "秋季": [13.52213184,17.17267811,19.28456754,25.89692456,23.364554,19.29411813,18.39706469,20.87623326]})

# TSM
# data = pd.DataFrame({"冬季": [61.33942848,53.06545816,66.16018363,59.75492281,56.13947823,39.95661586,39.74872857,26.54867376],
#                      "春季": [57.85237926,88.52542273,69.83858729,100.7954867,90.05153574,77.72663986,53.1615178,56.5886519],
#                      "夏季": [113.8424834,84.10512519,87.2216012,68.20572324,74.35835479,66.9090092,77.60736787,114.7180761],
#                      "秋季": [58.61171097,62.03683565,72.26684726,74.73771652,63.41899826,46.7319264,49.79716516,97.77236206]})


print(data)
std_table = data.std()  # 计算标准差
figdata = data.mean()  # 计算均值
print(figdata)
print(std_table)
# # 绘图
sns.set(font_scale=1.2)
plt.rc('font',family=['Times New Roman', 'SimSun'], size=12)
plt.errorbar(list(data.columns), figdata, yerr=std_table, fmt='k-o', lw=2, ecolor='k',elinewidth=1, ms=7, capsize=3, label='Chl-a季节平均浓度')
for a, b, c in zip(list(data.columns), figdata, std_table):
    plt.text(a, b+0.5, f'{round(b, 2)}±{round(c, 2)}', ha='center', va='bottom')
plt.xlabel('季节')
plt.ylabel('Chl-a浓度（ug/L）')
plt.legend()
plt.tight_layout()
plt.savefig('Chl-a.jpg', bbox_inches='tight', dpi=600)
plt.show()
