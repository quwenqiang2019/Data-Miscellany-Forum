from scipy import stats

# 两组样本数据
group1 = [1, 2, 3, 4, 5]
group2 = [2, 4, 6, 8, 10]

# 执行 T 检验
t_statistic, p_value = stats.ttest_ind(group1, group2)

# 输出结果
print("T statistic:", t_statistic)
print("P-value:", p_value)