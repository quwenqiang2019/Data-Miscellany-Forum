import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='once')

df = pd.read_csv("C:\工作\学习\数据杂坛/datasets/mtcars.csv")
x = df.loc[:, ['mpg']]
df['mpg_z'] = (x - x.mean()) / x.std()
df['colors'] = ['red' if x < 0 else 'green' for x in df['mpg_z']]
df.sort_values('mpg_z', inplace=True)
df.reset_index(inplace=True)

# Draw plot
plt.figure(figsize=(10, 6), dpi=80)
plt.hlines(y=df.index,
           xmin=0,
           xmax=df.mpg_z,
           color=df.colors,
           alpha=0.8,
           linewidth=5)
           
for x, y, tex in zip(df.mpg_z, df.index, df.mpg_z):
    t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left', 
                 verticalalignment='center', fontdict={'color':'black' if x < 0 else 'black', 'size':10})

# Decorations
plt.gca().set(ylabel='$Model', xlabel='$Mileage')
plt.yticks(df.index, df.cars, fontsize=12)
plt.xticks(fontsize=12)
plt.title('Diverging Bars of Car Mileage')
plt.grid(linestyle='--', alpha=0.5)
plt.show()