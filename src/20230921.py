import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
df = pd.DataFrame(X, columns=range(30))
df['y'] = y

correlations = df.corrwith(df.y).abs()
correlations.sort_values(ascending=False, inplace=True)

correlations.plot.bar()
plt.style.use('ggplot')
plt.figure(figsize=(10, 8))
plt.show()