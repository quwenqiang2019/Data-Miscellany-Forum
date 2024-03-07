import numpy as np
import matplotlib.pyplot as plt
# 用于正常显示中文
plt.rcParams['font.sans-serif'] = 'SimHei'
#用于正常显示符号
plt.rcParams['axes.unicode_minus'] = False

# 使用ggplot的绘图风格，这个类似于美化了，可以通过plt.style.available查看可选值，你会发现其它的风格真的丑。。。
plt.style.use('ggplot')

# 构造数据
values1 = [2.6,2.1,3.4,3,4.1]
values2 = [1.7,4.1,3.3,2.6,3.8]
feature = ['个人能力','QC知识','解决问题能力','服务质量意识','团队精神']

# 设置每个数据点的显示位置，在雷达图上用角度表示
angles=np.linspace(0, 2*np.pi,len(feature), endpoint=False)
angles=np.concatenate((angles,[angles[0]]))
feature = np.concatenate((feature, [feature[0]]))

# 绘图
fig=plt.figure()
for values in [values1, values2]:
# 拼接数据首尾，使图形中线条封闭
    values=np.concatenate((values,[values[0]]))
    # 设置为极坐标格式
    ax = fig.add_subplot(111, polar=True)
    # 绘制折线图
    ax.plot(angles, values, 'o-', linewidth=2)
    # 填充颜色
    ax.fill(angles, values, alpha=0.25)

# 设置图标上的角度划分刻度，为每个数据点处添加标签
ax.set_thetagrids(angles * 180/np.pi, feature)

# 设置雷达图的范围
ax.set_ylim(0,5)
# 添加标题
plt.title('活动前后员工状态表现')
# 添加网格线
ax.grid(True)

plt.show()