from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
import numpy as np



# =======================研究者招募30个学生参与一项实验。参与者被随机分配至三个组，每组使用不同学习方法准备三周后学习测试。三周后所有学生参加相同的测试。
# =======================研究三种学习方法在考试成绩上是否存在差异？我们首先提出一个原假设：各组均值相等。 让我们用 F 检验评估这个假设的合理性。=======================
# 加载每组成绩
group1 = [85, 86, 88, 75, 78, 94, 98, 79, 71, 80]  # 第一组学习方法，学生的测试成绩
group2 = [91, 92, 93, 85, 87, 84, 82, 88, 95, 96]  # 第二组学习方法，学生的测试成绩
group3 = [79, 78, 88, 94, 92, 85, 83, 85, 82, 81]  # 第三组学习方法，学生的测试成绩
# 执行 F 检验
f_statistic, p_value = f_oneway(group1, group2, group3)
# 打印结果
print("F 统计量:", f_statistic)
print("p 值:", p_value)




# ===========chi2_contingency卡方检验简介用法示例详解简介=====================
# =============,是否有证据表明阿司匹林降低了缺血性中风的风险？我们首先提出一个原假设： 阿司匹林的效果与安慰剂相当。 让我们用卡方检验评估这个假设的合理性。
sample1 = [176, 230]   # 第一组缺血性中风，使用阿司匹林和安慰剂的人数
sample2 = [21035, 21018]   # 第二组无中风，使用阿司匹林和安慰剂的人数

# 执行卡方检验
statistic, p_value, dof, expected = chi2_contingency(np.array([sample1, sample2]))
# 打印结果
print("Chi-square statistic:", statistic)
print("p-value:", p_value)
print("Degrees of freedom:", dof)
print("Expected frequencies:", expected)