from scipy.stats import f_oneway

# 每个样本的数据
sample1 = [1, 2, 3, 4, 5]
sample2 = [2, 4, 6, 8, 10]
sample3 = [3, 6, 9, 12, 15]

# 执行 F 检验
f_statistic, p_value = f_oneway(sample1, sample2, sample3)

# 打印结果
print("F 统计量:", f_statistic)
print("p 值:", p_value)