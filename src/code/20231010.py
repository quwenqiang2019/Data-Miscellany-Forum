from scipy.stats import chi2_contingency

# 例1：构建列联表（二维数组）
observed = [[37,27],
            [39,21]]

# 执行卡方检验
statistic, p_value, dof, expected = chi2_contingency(observed)

# 打印结果
print("Chi-square statistic:", statistic)
print("p-value:", p_value)
print("Degrees of freedom:", dof)
print("Expected frequencies:", expected)


# 例2：构建列联表（二维数组）
observed = [[11.7,8.7,15.4,8.4],
            [18.1,11.7,24.3,13.6],
            [26.9,20.3,37,19.3],
            [41,30.9,54.6,35.1],
            [66,54.3,71.1,50]]

# 执行扩展卡方检验
statistic, p_value, dof, expected = chi2_contingency(observed)

# 打印结果
print("Chi-square statistic:", statistic)
print("p-value:", p_value)
print("Degrees of freedom:", dof)
print("Expected frequencies:", expected)
