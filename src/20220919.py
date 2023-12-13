import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='once')

# Import Data
df = pd.read_csv("C:/工作/学习/数据杂坛/datasets/mpg_ggplot2.csv")

# Draw Plot
plt.figure(figsize=(10, 8), dpi=80)
sns.distplot(df.loc[df['class'] == 'compact', "cty"],
             color="#01a2d9",
             label="Compact",
             hist_kws={'alpha': .7},
             kde_kws={'linewidth': 3})
sns.distplot(df.loc[df['class'] == 'suv', "cty"],
             color="#dc2624",
             label="SUV",
             hist_kws={'alpha': .7},
             kde_kws={'linewidth': 3})
sns.distplot(df.loc[df['class'] == 'minivan', "cty"],
             color="g",
             label="minivan",
             hist_kws={'alpha': .7},
             kde_kws={'linewidth': 3})
plt.ylim(0, 0.35)

# Decoration
# font = {'family':  'Fangsong','size': 14,}
# sns.set(font_scale=1.2)
# plt.rc('font', family='Fangsong')
# mpl.rcParams["axes.unicode_minus"] = False
sns.set(font_scale=1.2)
plt.title('Density Plot of City Mileage by Vehicle Type', fontsize=18)
plt.legend()
plt.savefig('C:\工作\学习\数据杂坛\素材\\0919\密度图', dpi=300, bbox_inches = 'tight')
plt.show()
