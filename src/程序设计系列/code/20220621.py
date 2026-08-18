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

def draw_Marginal_Histogram(file):
    # Import Data
    df = pd.read_csv(file)

    # Create Fig and gridspec
    fig = plt.figure(figsize=(10, 6), dpi=100)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

    # Define the axes
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

    # Scatterplot on main ax
    ax_main.scatter('displ',
                    'hwy',
                    s=df.cty * 4,
                    c=df.manufacturer.astype('category').cat.codes,
                    alpha=.9,
                    data=df,
                    cmap="Set1",
                    edgecolors='gray',
                    linewidths=.5)

    # histogram on the right
    ax_bottom.hist(df.displ,
                   40,
                   histtype='stepfilled',
                   orientation='vertical',
                   color='#098154')
    ax_bottom.invert_yaxis()

    # histogram in the bottom
    ax_right.hist(df.hwy,
                  40,
                  histtype='stepfilled',
                  orientation='horizontal',
                  color='#098154')

    # Decorations
    ax_main.set(title='Scatterplot with Histograms \n displ vs hwy',
                xlabel='displ',
                ylabel='hwy')
    ax_main.title.set_fontsize(10)
    for item in ([ax_main.xaxis.label, ax_main.yaxis.label] +
                 ax_main.get_xticklabels() + ax_main.get_yticklabels()):
        item.set_fontsize(10)

    xlabels = ax_main.get_xticks().tolist()
    ax_main.set_xticklabels(xlabels)
    plt.show()

draw_Marginal_Histogram("F:\数据杂坛\datasets\mpg_ggplot2.csv")
