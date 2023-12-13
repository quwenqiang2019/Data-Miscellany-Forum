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

def draw_Marginal_Boxplot(file):
    # Import Data
    df = pd.read_csv(file)

    # Create Fig and gridspec
    fig = plt.figure(figsize=(10, 8), dpi=100)
    grid = plt.GridSpec(
        4, 4, hspace=0.5, wspace=0.2
    )  # 这里使用了matplotlib.pyplot.GridSpec分片figure

    # Define the axes
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

    # Scatterplot on main ax
    ax_main.scatter('displ',
                    'hwy',
                    s=df.cty * 5,
                    c=df.manufacturer.astype('category').cat.codes,
                    alpha=.9,
                    data=df,
                    cmap="Set1",
                    edgecolors='black',
                    linewidths=.5)

    # Add a graph in each part
    sns.boxplot(df.hwy, ax=ax_right, orient="v", linewidth=1, palette='Set1')
    sns.boxplot(df.displ, ax=ax_bottom, orient="h", linewidth=1, palette='Set1')

    # Decorations ------------------
    # Remove x axis name for the boxplot
    ax_bottom.set(xlabel='')
    ax_right.set(ylabel='')

    # Main Title, Xlabel and YLabel
    ax_main.set(title='Scatterplot with Histograms \n displ vs hwy',
                xlabel='displ',
                ylabel='hwy')

    # Set font size of different components
    ax_main.title.set_fontsize(12)
    for item in ([ax_main.xaxis.label, ax_main.yaxis.label] +
                 ax_main.get_xticklabels() + ax_main.get_yticklabels()):
        item.set_fontsize(11)

    plt.show()

draw_Marginal_Boxplot("F:\数据杂坛\datasets\mpg_ggplot2.csv")
