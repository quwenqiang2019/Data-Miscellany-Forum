import pandas as pd

data1 = {
    "a": [1, 2, 3, 12, 13, 14],
    "b": [4, 5, 6, 15, 16, 17],
    "c": [7, 8, 9, 18, 19, 20]
}
df = pd.DataFrame(data1)
print(df)
from sklearn.utils import shuffle
df = shuffle(df)
print(df)