import pandas as pd

data = pd.read_csv('Dataset.csv')
df = pd.DataFrame(data)
print(df.head())


index = df.index
print(index)
print(type(index))
print(list(index))


columns = df.columns
print(columns)
print(type(columns))
print(list(columns))


values = df.values
print(values)
print(type(values))


