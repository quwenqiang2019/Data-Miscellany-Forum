from scipy.stats import mannwhitneyu

# 两个独立样本的数据
group1 = [1, 2, 3, 4, 5]
group2 = [6, 7, 8, 9, 10]

# 执行Mann-Whitney U检验
statistic, p_value = mannwhitneyu(group1, group2)

# 打印结果
print("Mann-Whitney U statistic:", statistic)
print("p-value:", p_value)