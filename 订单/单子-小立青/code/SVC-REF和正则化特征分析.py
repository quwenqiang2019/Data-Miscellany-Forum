import os
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import numpy as np

# =====================读取数据=========================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
data = pd.read_csv(os.path.join(base_dir, 'data', 'heartbalance.csv'))
df = pd.DataFrame(data)
target = 'target'
features = df.columns.drop(target)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# =====================ref（svc）特征重要性分析=========================
model = SVC(kernel="linear", C=1)
selector = RFE(estimator=model, n_features_to_select=1, step=1)
selector.fit(X_train, y_train)
feature_ranking = selector.ranking_
ranking_df = pd.DataFrame({'Feature': features, 'Ranking': feature_ranking})
ranking_df = ranking_df.sort_values(by='Ranking')
sns.set(font_scale=1.2)
plt.rc('font',family=['Times New Roman', 'SimSun'], size=12)
plt.figure(figsize=(20, 16))
sns.barplot(x='Ranking', y='Feature', data=ranking_df)
plt.title('Feature Ranking from Linear SVC')
plt.xlabel('Ranking')
plt.ylabel('Feature')
plt.savefig(os.path.join(base_dir, 'result', 'ref_svc.jpg'), bbox_inches='tight', dpi = 600)
plt.show()

## ==================================svc 正则化================================
lsvc = LinearSVC(penalty= 'l1', C = 0.1,dual=False).fit(X_train, y_train)
model = SelectFromModel(estimator = lsvc, prefit=True,max_features=10)
X_selected = model.transform(X_train)
selected_features = model.get_support(indices=True)
print("Selected feature indices:", selected_features)
feature_names = features[selected_features]
print("Selected feature names:", feature_names)
coef = np.abs(lsvc.coef_).flatten()
ranking_df = pd.DataFrame({'Feature': features, 'Ranking': coef})
ranking_df = ranking_df.sort_values(by='Ranking')
sns.set(font_scale=1.2)
plt.rc('font',family=['Times New Roman', 'SimSun'], size=12)
plt.figure(figsize=(10, 10))
sns.barplot(x='Ranking', y='Feature', data=ranking_df)
plt.xlabel("Feature importance (absolute coefficient)")
plt.title("Feature importance from LinearSVC")
plt.savefig(os.path.join(base_dir, 'result', 'lsvc.jpg'), bbox_inches='tight', dpi = 600)
plt.show()



