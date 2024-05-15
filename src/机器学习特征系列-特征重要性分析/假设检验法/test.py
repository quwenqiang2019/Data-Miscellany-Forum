from scipy.stats import chi2_contingency
from scipy.stats import f_oneway


sample1 = [1, 2, 3, 4, 5]   # 第一组低收入人群（5个样本量），家庭车辆数
sample2 = [2, 4, 6, 8]   # 第一组低收入人群（4个样本量），家庭车辆数
sample3 = [3, 6, 9, 12, 15, 20]  # 第一组低收入人群（6个样本量），家庭车辆数
# 执行 F 检验
f_statistic, p_value = f_oneway(sample1, sample2, sample3)
# 打印结果
print("F 统计量:", f_statistic)
print("p 值:", p_value)


sample1 = [1, 2, 1, 1, 2]   # 第一组低收入人群（5个样本量），对生2胎的看法，3：支持，2：随意，1：反对
sample2 = [2, 2, 1, 2]   # 第一组低收入人群（4个样本量），对生2胎的看法，3：支持，2：随意，1：反对
sample3 = [1, 2, 1, 3, 1, 3]  # 第一组低收入人群（6个样本量），对生2胎的看法，3：支持，2：随意，1：反对
# 执行卡方检验
statistic, p_value, dof, expected = chi2_contingency(sample1, sample2, sample3)
# 打印结果
print("Chi-square statistic:", statistic)
print("p-value:", p_value)
print("Degrees of freedom:", dof)
print("Expected frequencies:", expected)