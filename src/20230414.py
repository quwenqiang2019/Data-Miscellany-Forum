import pandas as pd

data = pd.DataFrame()
a = [[1, 2, 3], [4, 5, 6]]
data = data.append(a)
a = [[7, 8, 9], [10, 11, 12]]
data = data.append(a)
print(data)

# index不出现重复情况，设置ignore_index=True
data = pd.DataFrame()
a = [[1, 2, 3], [4, 5, 6]]
data = data.append(a, ignore_index=True)
a = [[7, 8, 9], [10, 11, 12]]
data = data.append(a, ignore_index=True)
print(data)