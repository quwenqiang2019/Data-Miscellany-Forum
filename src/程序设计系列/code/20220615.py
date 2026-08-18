import numpy as np
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
    # Import dataset
    midwest = pd.read_csv(file)
    # Prepare Data
    # Create as many colors as there are unique midwest['category']
    categories = np.unique(midwest['category'])
    colors = [plt.cm.Set1(i / float(len(categories) - 1)) for i in range(len(categories))]
    # Draw Plot for Each Category
    plt.figure(figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')

    for i, category in enumerate(categories):
        plt.scatter('area', 'poptotal', data=midwest.loc[midwest.category == category, :],s=20,c=colors[i],label=str(category))

    # Decorations
    plt.gca().set(xlim=(0.0, 0.1), ylim=(0, 90000),)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Area', fontdict={'fontsize': 10})
    plt.ylabel('Population', fontdict={'fontsize': 10})
    plt.title("Scatterplot of Midwest Area vs Population", fontsize=12)
    plt.legend(fontsize=10)
    plt.show()

draw_scatter("F:\数据杂坛\datasets\midwest_filter.csv")
