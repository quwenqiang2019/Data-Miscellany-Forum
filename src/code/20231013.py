from scipy.stats import kruskal

# 多组独立样本的数据
group1 = [1, 2, 3, 4, 5]
group2 = [6, 7, 8, 9, 10]
group3 = [11, 12, 13, 14, 15]

# 执行Kruskal-Wallis H检验
statistic, p_value = kruskal(group1, group2, group3)

# 打印结果
print("Kruskal-Wallis H statistic:", statistic)
print("p-value:", p_value)