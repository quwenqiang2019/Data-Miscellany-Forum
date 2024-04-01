import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

font = {'family': 'Times New Roman',
        'size': 12,
        }
sns.set(font_scale=1.2)
plt.rc('font',family='Times New Roman')
plt.style.use('ggplot')# 使用ggplot的绘图风格

# 构造数据
values1= [0.778, 0.833, 0.818, 0.847]
values2= [0.818, 0.846, 0.833, 0.881]
values3= [0.909, 0.846, 0.875, 0.850]

feature = ["Sensitivity","Specificity","Accuracy","AUC"]

# 设置每个数据点的显示位置，在雷达图上用角度表示
angles=np.linspace(0, 2*np.pi,len(feature), endpoint=False)
angles=np.concatenate((angles,[angles[0]]))
feature = np.concatenate((feature, [feature[0]]))

# 绘图
fig=plt.figure(figsize=(8,8))
# 设置为极坐标格式
ax = fig.add_subplot(111, polar=True)

for values in [values1, values2,values3]:
# 拼接数据首尾，使图形中线条封闭
    values=np.concatenate((values,[values[0]]))
    # 绘制折线图
    ax.plot(angles, values, 'o-', linewidth=2)

for values in [values1, values2,values3]:
    values=np.concatenate((values,[values[0]]))
    # 填充颜色
    ax.fill(angles, values, alpha=0.25)

# 设置图标上的角度划分刻度，为每个数据点处添加标签
ax.set_thetagrids(angles * 180/np.pi, feature,fontsize=14,style='italic')
# 设置雷达图的范围
ax.set_ylim(0.5,1)
# 设置雷达图的0度起始位置
ax.set_theta_zero_location('N')
# 设置雷达图的坐标值显示角度，相对于起始角度的偏移量
ax.set_rlabel_position(270)
plt.legend(["Model1", "Model2",'Model3'], loc='best')
# 添加标题
plt.title('Comparison of classifier evaluation indicators',fontsize = 14)
# 添加网格线
plt.show()