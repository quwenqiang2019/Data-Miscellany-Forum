import numpy as np
import pandas as pd
from itertools import permutations
from itertools import combinations

array_2d = np.random.rand(4, 5)
print(array_2d)


B_1 = []
B_2 = []
dif = []
parameter = [1.2, 1, 0.8, 2, 0.3]
for pair in permutations(enumerate(array_2d), 2):
    idx_a, array_a= pair[0]
    idx_b, array_b = pair[1]
    diff = array_a - array_b
    diff_corr = pd.Series(parameter).corr(pd.Series(diff))
    print(idx_a, idx_b, diff_corr)
    B_1.append(idx_a)
    B_2.append(idx_b)
    dif.append(diff_corr)

df = pd.DataFrame({'波段1':B_1, '波段2':B_2, '差值相关系数':dif})
df = df.sort_values(by='差值相关系数', key=abs)
print(df)