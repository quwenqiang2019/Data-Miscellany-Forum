import pandas as pd
import numpy as np

data = pd.DataFrame({
    'name': ['Bob', 'Mary', 'Peter', 'nancy', 'Lucy'],
    'score': [99, 100, np.nan, 91, 95],
    'class': ['class1', 'class2', 'class1', 'class2', np.nan],
    'sex': ['male', 'fmale', 'male', 'male', 'fmale'],
    'age': [23, 25, np.nan, 19, 24]
})

data[['score', 'age']] = data[['score', 'age']].fillna(data[['score', 'age']].mean())
data['class'] = data['class'].fillna(data['class'].mode()[0])

print(data)
