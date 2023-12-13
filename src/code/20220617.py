import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='once')
plt.style.use('seaborn-whitegrid')
sns.set_style("whitegrid")
print(mpl.__version__)
print(sns.__version__)

def draw_scatter(file):
    # Import Data
    df = pd.read_csv(file)
    df_select = df.loc[df.cyl.isin([4, 8]), :]

    # Plot
    gridobj = sns.lmplot(
        x="displ",
        y="hwy",
        hue="cyl",
        data=df_select,
        height=7,
        aspect=1.6,
        palette='Set1',
        scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))

    # Decorations
    sns.set(style="whitegrid", font_scale=1.5)
    gridobj.set(xlim=(0.5, 7.5), ylim=(10, 50))
    gridobj.fig.set_size_inches(10, 6)
    plt.tight_layout()
    plt.title("Scatterplot with line of best fit grouped by number of cylinders")
    plt.show()

draw_scatter("F:\数据杂坛\datasets\mpg_ggplot2.csv")
