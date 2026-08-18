import matplotlib.pyplot as plt
import seaborn as sns

# ==========================设置绘图风格===========================
# 法1：
# sns.set(style='darkgrid', font_scale=1.2)
# 法2：
plt.style.use('seaborn-darkgrid')

# ============设置字体，中文为SimSun，英文为Times New Roman========
# 法1：
plt.rcParams['font.family'] = 'Times New Roman, SimSun'
# 法2：
# font1 = {'family': 'Times New Roman, SimSun'}
# plt.rc('font', **font1)

# 法3：
# config = {
#     "font.family": 'Times New Roman, SimSun', # 衬线字体
#     "font.size": 12, # 相当于小四大小
#     # "font.serif": ['Times New Roman', 'SimSun'], # 宋体
#     "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
#     'axes.unicode_minus': False # 处理负号，即-号
# }
# plt.rcParams.update(config)

# =========================数据==============================
x = ['A', 'B', 'C', 'D']
y = [10, 20, 15, 25]

# =============================绘制柱状图====================
plt.bar(x, y)

# ===========================添加标题和标签===================
plt.title('柱状图示例',fontsize=16)
plt.xlabel('类别',fontsize=16)
plt.ylabel('数值',fontsize=16)

# ===============================显示图形===================
plt.show()
