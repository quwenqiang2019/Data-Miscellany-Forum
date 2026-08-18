import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve

# 准备数据
data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)
print(df)
# 提取目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
X = df[features]
y = df[target]

# 绘制学习曲线
train_sizes, train_scores, test_scores = learning_curve(SVC(gamma=0.01), X, y, cv=10, train_sizes=[0.1,0.25,0.5,0.75,1])
print('train_sizes:', train_sizes, sep='\n')
print('train_scores:', train_scores, sep='\n')
print('test_scores:', test_scores, sep='\n')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Validation")

plt.xlabel("Training examples")
plt.ylabel("scores")
plt.legend(loc="best")
plt.show()
