import pandas as pd
import numpy as np
df = pd.DataFrame({
    'group': [1, 1, 2, 3, 3, 3, 4],
    'param': ['a', 'a', 'b', np.nan, 'a', 'a', np.nan]
})
print(df)

print (df.groupby('param')['group'].nunique())

