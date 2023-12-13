import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.spatial import ConvexHull
warnings.filterwarnings(action='once')
plt.style.use('seaborn-whitegrid')
sns.set_style("whitegrid")
print(mpl.__version__)
print(sns.__version__)

def draw_scatter(file):
    # Step 1: Prepare Data
    midwest = pd.read_csv(file)

    # As many colors as there are unique midwest['category']
    categories = np.unique(midwest['category'])
    colors = [plt.cm.Set1(i / float(len(categories) - 1)) for i in range(len(categories))]

    # Step 2: Draw Scatterplot with unique color for each category
    fig = plt.figure(figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

    for i, category in enumerate(categories):
        plt.scatter('area','poptotal',data=midwest.loc[midwest.category == category, :],s='dot_size',c=colors[i],label=str(category),edgecolors='black',linewidths=.5)

    # Step 3: Encircling
    # https://stackoverflow.com/questions/44575681/how-do-i-encircle-different-data-sets-in-scatter-plot
    def encircle(x, y, ax=None, **kw):  # 定义encircle函数，圈出重点关注的点
        if not ax: ax = plt.gca()
        p = np.c_[x, y]
        hull = ConvexHull(p)
        poly = plt.Polygon(p[hull.vertices, :], **kw)
        ax.add_patch(poly)

    # Select data to be encircled
    midwest_encircle_data1 = midwest.loc[midwest.state == 'IN', :]
    encircle(midwest_encircle_data1.area,midwest_encircle_data1.poptotal,ec="pink",fc="#74C476",alpha=0.3)
    encircle(midwest_encircle_data1.area,midwest_encircle_data1.poptotal,ec="g",fc="none",linewidth=1.5)

    midwest_encircle_data6 = midwest.loc[midwest.state == 'WI', :]
    encircle(midwest_encircle_data6.area,midwest_encircle_data6.poptotal,ec="pink",fc="black",alpha=0.3)
    encircle(midwest_encircle_data6.area,midwest_encircle_data6.poptotal,ec="black",fc="none",linewidth=1.5,linestyle='--')

    # Step 4: Decorations
    plt.gca().set(xlim=(0.0, 0.1),ylim=(0, 90000),)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Area', fontdict={'fontsize': 14})
    plt.ylabel('Population', fontdict={'fontsize': 14})
    plt.title("Bubble Plot with Encircling", fontsize=14)
    plt.legend(fontsize=10)
    plt.show()

draw_scatter("F:\数据杂坛\datasets\midwest_filter.csv")
